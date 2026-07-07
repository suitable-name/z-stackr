use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use image::DynamicImage;

use stacker_algo::hybrid::retouch::RetouchSession;

use crate::{
    image_utils::planar_to_rgb_image,
    retouch::RetouchState,
    settings::{self, StackingSettings},
};

/// Saves the currently-previewed result image to disk, prompting the user
/// with a native save dialog.
///
/// When `forced_format` is `Some`, that format is offered as the
/// preferred/default format regardless of the configured
/// `image_saving.output_format` setting — this powers the "Save as PNG /
/// TIFF / JPG" context-menu shortcuts. The plain "Save" button passes `None`
/// to keep using the configured default format.
///
/// When a retouch session is active, the freshly-rendered brush composite
/// is saved instead of re-reading the on-disk temp file — otherwise brush
/// strokes would be display-only and silently vanish from the saved output.
///
/// # Panics
///
/// Calls `.lock().unwrap()` on the shared `Mutex`-guarded state (`viewed`,
/// `file_paths_s`, `settings_arc_s`, `retouch_arc_s`); those panic only if
/// another thread already panicked while holding the same lock (mutex
/// poisoning), which does not happen in normal operation.
// `too_many_lines`: one cohesive "resolve format -> build dialog -> resolve
// dest extension -> build the image -> encode -> copy metadata" save
// pipeline; splitting it would scatter state (the resolved `output_format`,
// `dest_path`, `src_img`) that each step depends on across artificial
// function boundaries.
#[allow(clippy::too_many_lines)]
pub fn perform_save_current_image(
    viewed: &Arc<Mutex<Option<PathBuf>>>,
    file_paths_s: &Arc<Mutex<Vec<PathBuf>>>,
    settings_arc_s: &Arc<Mutex<StackingSettings>>,
    retouch_arc_s: &Arc<Mutex<RetouchState>>,
    forced_format: Option<settings::OutputFormat>,
) {
    let src_path = viewed.lock().unwrap().clone();
    let save_cfg = settings_arc_s.lock().unwrap().image_saving.clone();

    let Some(src_path) = src_path else {
        return;
    };

    let output_format = forced_format.unwrap_or(save_cfg.output_format);

    // ── Derive suggested filename from template ────────────────────
    // Supported tokens: {name} → stem of the first source file.
    // {algo} is intentionally left unexpanded here (the save handler
    // doesn't know which algorithm produced the current result; the
    // temp filename carries that info but we keep the dialog clean).
    let first_stem = {
        let guard = file_paths_s.lock().unwrap();
        guard.first().and_then(|p| p.file_stem()).map_or_else(
            || "stacked".to_owned(),
            |s| s.to_string_lossy().into_owned(),
        )
    };
    let suggested_stem = save_cfg.filename_template.replace("{name}", &first_stem);

    // ── Extension / format filters (preferred format first) ────────
    let (default_ext, format_filters): (&str, &[(&str, &[&str])]) = match output_format {
        settings::OutputFormat::Png => (
            "png",
            &[
                ("PNG Image", &["png"]),
                ("TIFF Image", &["tif", "tiff"]),
                ("JPEG Image", &["jpg", "jpeg"]),
            ],
        ),
        settings::OutputFormat::Jpeg => (
            "jpg",
            &[
                ("JPEG Image", &["jpg", "jpeg"]),
                ("PNG Image", &["png"]),
                ("TIFF Image", &["tif", "tiff"]),
            ],
        ),
        settings::OutputFormat::Tiff => (
            "tiff",
            &[
                ("TIFF Image", &["tif", "tiff"]),
                ("PNG Image", &["png"]),
                ("JPEG Image", &["jpg", "jpeg"]),
            ],
        ),
    };
    let suggested_name = format!("{suggested_stem}.{default_ext}");

    // ── Build the save dialog ──────────────────────────────────────
    let mut dialog = rfd::FileDialog::new().set_file_name(suggested_name.as_str());

    // Apply default output directory when configured.
    if !save_cfg.default_output_dir.is_empty() {
        let dir = std::path::Path::new(&save_cfg.default_output_dir);
        if dir.is_dir() {
            dialog = dialog.set_directory(dir);
        }
    }

    // Add format filters (preferred format first).
    for (label, exts) in format_filters {
        dialog = dialog.add_filter(*label, exts);
    }

    let Some(dest_path) = dialog.save_file() else {
        return;
    };

    // ── Determine actual format from chosen extension ──────────────
    // If the user typed a recognised extension, honour it; otherwise
    // fall back to the preferred format.
    let chosen_ext = dest_path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    let effective_format = match chosen_ext.as_deref() {
        Some("png") => settings::OutputFormat::Png,
        Some("jpg" | "jpeg") => settings::OutputFormat::Jpeg,
        Some("tif" | "tiff") => settings::OutputFormat::Tiff,
        _ => output_format,
    };

    // ── Build the image to re-encode ────────────────────────────────
    // Prefer the live retouch composite (base/src blended by the current
    // brush alpha mask) over the on-disk temp file, so brush strokes are
    // actually reflected in the saved output — but only when the active
    // session actually belongs to the file being saved (`result_path`
    // matches `src_path`). This is the safety check that lets the retouch
    // popup's session survive freely browsing source frames/other results
    // in the main window (see `callbacks::files::on_file_clicked` and
    // `on_result_clicked`, which no longer wipe the session on every
    // click): instead of defensively clearing the session the moment the
    // user looks at something else, we simply never apply it here unless
    // it's actually for the file being saved right now.
    let retouched = {
        let rs = retouch_arc_s.lock().unwrap();
        (rs.result_path.as_deref() == Some(src_path.as_path()))
            .then(|| rs.session.as_ref().map(RetouchSession::render_composite))
            .flatten()
    };
    let src_img = match retouched {
        Some(composite) => DynamicImage::ImageRgb8(planar_to_rgb_image(&composite)),
        None => match image::open(&src_path) {
            Ok(img) => img,
            Err(e) => {
                tracing::error!("could not open temp result for re-encode: {e}");
                return;
            }
        },
    };

    // copy_metadata is handled after a successful save, below, via
    // `stacker_core::metadata::copy_metadata` (shared with the CLI).
    // It re-injects the raw EXIF blob from the first source frame
    // into the JPEG/PNG output at the byte level; TIFF output isn't
    // supported (see the module docs on `stacker_core::metadata`).
    let save_result = match effective_format {
        settings::OutputFormat::Tiff => {
            // Honour bit_depth for TIFF: 16-bit when configured, else 8-bit.
            if save_cfg.bit_depth == 16 {
                let rgb16 = src_img.to_rgb16();
                rgb16.save(&dest_path)
            } else {
                let rgb8 = src_img.to_rgb8();
                rgb8.save(&dest_path)
            }
        }
        settings::OutputFormat::Png => {
            // Honour bit_depth for PNG: 16-bit when configured, else 8-bit.
            if save_cfg.bit_depth == 16 {
                let rgb16 = src_img.to_rgb16();
                rgb16.save(&dest_path)
            } else {
                let rgb8 = src_img.to_rgb8();
                rgb8.save(&dest_path)
            }
        }
        settings::OutputFormat::Jpeg => {
            // JPEG is always 8-bit; bit_depth is ignored.
            use std::{fs::File, io::BufWriter};

            let rgb8 = src_img.to_rgb8();
            let file = match File::create(&dest_path) {
                Ok(f) => f,
                Err(e) => {
                    tracing::error!("could not create output file {}: {e}", dest_path.display());
                    return;
                }
            };

            // 1. Declare the writer as mutable
            let mut writer = BufWriter::new(file);

            // 2. Pass a mutable reference (&mut writer) to the encoder
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(
                &mut writer,
                save_cfg.jpeg_quality.clamp(1, 100) as u8,
            );

            // 3. Perform the encode
            if let Err(e) = enc.encode_image(&rgb8) {
                tracing::error!("failed to encode JPEG: {e}");
            }

            // Return the success unit literal
            Ok(())
        }
    };

    match save_result {
        Ok(()) => {
            tracing::info!("saved result to {}", dest_path.display());
            if save_cfg.copy_metadata {
                let first_source = file_paths_s.lock().unwrap().first().cloned();
                if let Some(first_source) = first_source {
                    match stacker_core::metadata::copy_metadata(&first_source, &dest_path) {
                        Ok(true) => tracing::info!(
                            source = %first_source.display(),
                            "copied EXIF metadata to output"
                        ),
                        Ok(false) => tracing::debug!(
                            source = %first_source.display(),
                            "copy_metadata enabled, but source frame has no EXIF metadata; nothing to copy"
                        ),
                        Err(e) => tracing::warn!("copy_metadata enabled but failed: {e}"),
                    }
                }
            }
        }
        Err(e) => tracing::error!("save failed for {}: {e}", dest_path.display()),
    }
}
