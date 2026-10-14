//! Output-path resolution: from `--output-file` and the `image_saving.*`
//! settings to the file, format, bit depth and JPEG quality of the result.
//!
//! The pixels themselves are encoded and written by the shared
//! `stacker_pipeline::output::EncodedImage`; the full-image buffer it holds
//! (3 bytes per pixel at 8 bits, 6 at 16) is the only full-image allocation of
//! the tiled path.
use stacker_core::{
    error::StackerError,
    settings::{ImageSavingSettings, OutputFormat},
};
use stacker_pipeline::output::{jpeg_quality, wants_16_bit};
use std::path::{Path, PathBuf};

const fn invalid_input(msg: String) -> StackerError {
    StackerError::Validation(msg)
}

/// File extension (without the dot) the configured format is written with.
#[must_use]
pub const fn extension_for(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Tiff => "tiff",
        OutputFormat::Png => "png",
        OutputFormat::Jpeg => "jpg",
    }
}

/// File name from `filename_template` (`{name}` = `first_stem`) inside `dir`.
///
/// # Errors
/// [`StackerError::Validation`] when the template produces an empty file name.
pub fn templated_file_in(
    dir: &Path,
    saving: &ImageSavingSettings,
    first_stem: &str,
) -> Result<PathBuf, StackerError> {
    let stem = saving.filename_template.replace("{name}", first_stem);
    if stem.trim().is_empty() {
        return Err(invalid_input(
            "image_saving.filename_template produces an empty file name".to_owned(),
        ));
    }
    Ok(dir.join(format!("{stem}.{}", extension_for(saving.output_format))))
}

/// A user-supplied file name: the configured extension is appended when it has
/// none, and a bare name is placed in `default_output_dir` when that is set.
fn named_file(output_file: &Path, saving: &ImageSavingSettings) -> Result<PathBuf, StackerError> {
    if output_file.file_name().is_none() {
        return Err(invalid_input(format!(
            "--output-file '{}' has no file name",
            output_file.display()
        )));
    }
    let mut path = output_file.to_path_buf();
    if path.extension().is_none() {
        path.set_extension(extension_for(saving.output_format));
    }
    let bare = path.parent().is_none_or(|p| p.as_os_str().is_empty());
    if bare && !saving.default_output_dir.is_empty() {
        path = Path::new(&saving.default_output_dir).join(&path);
    }
    Ok(path)
}

/// Where and how the stacked image is written, resolved from `--output-file`
/// and the `image_saving.*` settings before any frame is decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputPlan {
    /// Final file path (extension included).
    pub path: PathBuf,
    /// Container format; always agrees with the extension of `path`.
    pub format: OutputFormat,
    /// Write 16 bits per channel (PNG and TIFF only; JPEG is always 8-bit).
    pub bit16: bool,
    /// JPEG quality, 1..=100.
    pub jpeg_quality: u8,
}

/// Resolve the output file from `--output-file` and `image_saving.*`.
///
/// * `--output-file` is normally a complete file name; its extension chooses
///   the format (`.png`, `.tif`/`.tiff`, `.jpg`/`.jpeg`), as it does for the
///   GUI's save dialog.
/// * With no extension, the extension of `output_format` is appended.
/// * When it names an existing directory (or ends in a path separator), the
///   file name is built from `filename_template` (`{name}` = stem of the first
///   frame) plus the extension of `output_format`.
/// * A bare file name (no directory part) is placed in `default_output_dir`
///   when that is set.
/// * `bit_depth` selects 8 or 16 bits for PNG/TIFF; `jpeg_quality` the JPEG
///   quality.
///
/// # Errors
/// Returns an error for an unsupported extension or an empty file name.
pub fn resolve_output(
    output_file: &Path,
    saving: &ImageSavingSettings,
    first_stem: &str,
) -> Result<OutputPlan, StackerError> {
    let raw = output_file.to_string_lossy();
    let wants_directory = output_file.is_dir() || raw.ends_with('/') || raw.ends_with('\\');
    let path = if wants_directory {
        templated_file_in(output_file, saving, first_stem)?
    } else {
        named_file(output_file, saving)?
    };

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    let format = match ext.as_deref() {
        Some("png") => OutputFormat::Png,
        Some("jpg" | "jpeg") => OutputFormat::Jpeg,
        Some("tif" | "tiff") => OutputFormat::Tiff,
        other => {
            return Err(invalid_input(format!(
                "unsupported output extension {} in '{}'; use .png, .tif/.tiff or .jpg/.jpeg",
                other.map_or_else(|| "(none)".to_owned(), |e| format!("'.{e}'")),
                path.display()
            )));
        }
    };

    Ok(OutputPlan {
        path,
        format,
        bit16: wants_16_bit(format, saving.bit_depth),
        jpeg_quality: jpeg_quality(saving.jpeg_quality),
    })
}

/// Check that the planned output file can be written, creating its parent
/// directory if needed.  Runs before any frame is decoded, so a wrong path
/// fails in milliseconds instead of after the whole alignment and fusion.
///
/// An already existing file is left untouched; a probe file created by this
/// check is removed again.
///
/// # Errors
/// Returns an error when the parent is not a directory, the target is a
/// directory, or the file cannot be opened for writing.
pub fn check_output_writable(plan: &OutputPlan) -> Result<(), StackerError> {
    if let Some(parent) = plan.path.parent().filter(|p| !p.as_os_str().is_empty()) {
        if parent.exists() && !parent.is_dir() {
            return Err(invalid_input(format!(
                "output directory '{}' exists but is not a directory",
                parent.display()
            )));
        }
        std::fs::create_dir_all(parent).map_err(|e| {
            invalid_input(format!(
                "cannot create output directory '{}': {e}",
                parent.display()
            ))
        })?;
    }
    if plan.path.is_dir() {
        return Err(invalid_input(format!(
            "output path '{}' is a directory",
            plan.path.display()
        )));
    }
    let existed = plan.path.exists();
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&plan.path)
        .map_err(|e| {
            invalid_input(format!(
                "cannot write output file '{}': {e}",
                plan.path.display()
            ))
        })?;
    if !existed {
        let _ = std::fs::remove_file(&plan.path);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use stacker_core::image::PlanarImage;
    use stacker_pipeline::output::encode;

    fn saving() -> ImageSavingSettings {
        ImageSavingSettings::default()
    }

    #[test]
    fn extension_chooses_the_format_and_defaults_give_16_bit() {
        let plan = resolve_output(Path::new("out/result.PNG"), &saving(), "frame").unwrap();
        assert_eq!(plan.format, OutputFormat::Png);
        assert!(plan.bit16);

        let plan = resolve_output(Path::new("result.jpeg"), &saving(), "frame").unwrap();
        assert_eq!(plan.format, OutputFormat::Jpeg);
        assert!(!plan.bit16, "JPEG is always 8-bit");
        assert_eq!(plan.jpeg_quality, 95);
    }

    #[test]
    fn unsupported_extension_is_rejected_with_a_clear_message() {
        let err = resolve_output(Path::new("out.xyz"), &saving(), "frame").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(".xyz") && msg.contains("unsupported"), "{msg}");
    }

    #[test]
    fn saving_keys_shape_the_plan() {
        let mut s = saving();
        s.bit_depth = 8;
        s.jpeg_quality = 30;
        s.output_format = OutputFormat::Png;
        s.filename_template = "{name}_final".to_owned();
        s.default_output_dir = "results".to_owned();

        // No extension: the configured format supplies it; a bare name goes to
        // default_output_dir.
        let plan = resolve_output(Path::new("shot"), &s, "frame").unwrap();
        assert_eq!(plan.path, Path::new("results").join("shot.png"));
        assert!(!plan.bit16);

        // A name with a directory part is taken as given.
        let plan = resolve_output(Path::new("elsewhere/shot.jpg"), &s, "frame").unwrap();
        assert_eq!(plan.path, Path::new("elsewhere/shot.jpg"));
        assert_eq!(plan.jpeg_quality, 30);

        // A trailing separator asks for a templated name inside that directory.
        let plan = resolve_output(Path::new("dir/"), &s, "frame_0001").unwrap();
        assert_eq!(
            plan.path.file_name().unwrap().to_string_lossy(),
            "frame_0001_final.png"
        );
    }

    #[test]
    fn parent_that_is_a_file_is_rejected() {
        let file = std::env::temp_dir().join(format!("stacker_cli_notadir_{}", std::process::id()));
        std::fs::write(&file, b"x").unwrap();
        let plan = resolve_output(&file.join("out.png"), &saving(), "f").unwrap();
        let err = check_output_writable(&plan).unwrap_err();
        assert!(err.to_string().contains("not a directory"), "{err}");
        let _ = std::fs::remove_file(&file);
    }

    #[test]
    fn writable_check_leaves_no_probe_file_behind() {
        let dir = std::env::temp_dir().join(format!("stacker_cli_probe_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let plan = resolve_output(&dir.join("sub").join("out.png"), &saving(), "f").unwrap();
        check_output_writable(&plan).unwrap();
        assert!(!plan.path.exists());
        assert!(dir.join("sub").is_dir());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn jpeg_quality_changes_the_encoded_size() {
        let mut img = PlanarImage::<f32>::new(64, 64);
        for y in 0..64_usize {
            for x in 0..64_usize {
                let v = ((x * 37 + y * 91) ^ (x * y)) & 0xff;
                let i = y * 64 + x;
                img.luma[i] = f32::from(u8::try_from(v).unwrap()) / 255.0;
                img.chroma_a[i] = (img.luma[i] - 0.5) * 0.2;
                img.chroma_b[i] = (0.5 - img.luma[i]) * 0.2;
            }
        }
        let dir = std::env::temp_dir().join(format!("stacker_cli_jpeg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut sizes = Vec::new();
        for q in [30_u32, 95] {
            let mut s = saving();
            s.jpeg_quality = q;
            let plan = resolve_output(&dir.join(format!("q{q}.jpg")), &s, "f").unwrap();
            let mut saving = s;
            saving.output_format = plan.format;
            encode(&img, &saving).save(&plan.path).unwrap();
            sizes.push(std::fs::metadata(&plan.path).unwrap().len());
        }
        assert!(
            sizes[0] < sizes[1],
            "q30 {} must be smaller than q95 {}",
            sizes[0],
            sizes[1]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
