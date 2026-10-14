//! Export the image on the canvas: the export dialog, the file-name template and
//! the encode worker.
//!
//! Two entry points reach the same worker:
//!
//! * **Export…** (File menu, Ctrl+E, toolbar) opens the export dialog
//!   (`ui/components/export_dialog.slint`). Its **Save** button writes the file
//!   straight into the configured output folder under the name the template gives
//!   (a number is appended instead of overwriting). Without a folder, Save asks
//!   for the name in a save dialog, pre-filled from the template.
//! * **Save As…** (the dialog's button, `App.save-current-image`) always asks for
//!   the name in a save dialog.
//!
//! The dialog edits the folder, template and format through
//! `SettingsModel`, so they are the persisted settings. The template's variables
//! and the sanitising live in [`stacker_gui::export::filename_template`]; the
//! live preview answers from a [`TemplateContext`] cached when the dialog opens,
//! so the pure `preview-name` callback never touches the application state.
//!
//! The encode runs on a worker and reports the outcome through
//! [`super::feedback`]. What is written depends on what is on the canvas:
//!
//! * the **newest result**: the `f32` fused image composited with the retouch
//!   strokes, encoded at the configured bit depth (a 16-bit PNG or TIFF holds real
//!   16-bit samples);
//! * **anything else** (a source frame, an older result): the file itself,
//!   re-encoded. The toast says so, because only the newest result exists at full
//!   precision.

use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    rc::Rc,
};

use slint::{ComponentHandle, SharedString};
use stacker_gui::{
    export::{
        SavedAs, encode_image_bytes, encode_planar_exif, exif_block, exif_note,
        filename_template::{
            self as template, TemplateContext, VARIABLES, resolve_path, unknown_placeholders,
        },
        save_file, save_planar_exif,
    },
    latest_worker::spawn_or_inline,
    pickers::{PickerKind, PickerRequest},
    platform::Platform,
    settings::{ImageSavingSettings, OutputFormat, PreprocessingSettings},
    stall_guard::stall_guard,
    state::{AppState, SaveTarget},
};

use super::{
    Ctx,
    feedback::{self, MessageKind},
    pickers,
};
use crate::{App, ExportModel, ProjectModel};

type FormatFilters = &'static [(&'static str, &'static [&'static str])];

/// The project file suffix; stripped from the project name for `{name}`.
const PROJECT_SUFFIX: &str = ".zstack.toml";

/// The template context the dialog's live preview renders against; set whenever
/// the dialog opens.
type PreviewContext = Rc<RefCell<TemplateContext>>;

/// Register the export dialog's and Save As…'s callbacks.
pub fn setup(ui: &App, ctx: &Ctx) {
    let preview_context: PreviewContext = Rc::new(RefCell::new(TemplateContext::default()));
    let model = ui.global::<ExportModel>();
    model.set_placeholders(placeholder_help().into());

    let save_as_ctx = ctx.clone();
    ui.on_save_current_image(move || {
        stall_guard("export::save-current-image", || save_as(&save_as_ctx));
    });

    let open_ctx = ctx.clone();
    let open_cache = Rc::clone(&preview_context);
    model.on_request_open(move || {
        stall_guard("export::request-open", || {
            open_dialog(&open_ctx, &open_cache);
        });
    });

    let save_ctx = ctx.clone();
    model.on_save(move || {
        stall_guard("export::save", || save_with_template(&save_ctx));
    });

    let cache = Rc::clone(&preview_context);
    model.on_preview_name(move |text| {
        stall_guard("export::preview-name", || {
            SharedString::from(template::preview(text.as_str(), &cache.borrow()))
        })
    });
    model.on_unknown_placeholders(move |text| {
        stall_guard("export::unknown-placeholders", || {
            SharedString::from(unknown_placeholders(text.as_str()).join(", "))
        })
    });
}

/// The line under the template field: every variable, braced.
fn placeholder_help() -> String {
    let names: Vec<String> = VARIABLES.iter().map(|name| format!("{{{name}}}")).collect();
    format!("Variables: {}", names.join(" "))
}

/// Default extension and the dialog's filter list (preferred format first).
const fn format_filters(format: OutputFormat) -> (&'static str, FormatFilters) {
    match format {
        OutputFormat::Png => (
            "png",
            &[
                ("PNG Image", &["png"]),
                ("TIFF Image", &["tif", "tiff"]),
                ("JPEG Image", &["jpg", "jpeg"]),
            ],
        ),
        OutputFormat::Jpeg => (
            "jpg",
            &[
                ("JPEG Image", &["jpg", "jpeg"]),
                ("PNG Image", &["png"]),
                ("TIFF Image", &["tif", "tiff"]),
            ],
        ),
        OutputFormat::Tiff => (
            "tiff",
            &[
                ("TIFF Image", &["tif", "tiff"]),
                ("PNG Image", &["png"]),
                ("JPEG Image", &["jpg", "jpeg"]),
            ],
        ),
    }
}

/// The suggested file name for a save dialog: the template's stem plus the
/// format's extension.
fn suggested_name(cfg: &ImageSavingSettings, names: &TemplateContext) -> String {
    let (ext, _) = format_filters(cfg.output_format);
    format!("{}.{ext}", template::stem(&cfg.filename_template, names))
}

/// The configured default output directory, when it exists.
fn starting_dir(cfg: &ImageSavingSettings) -> Option<PathBuf> {
    if cfg.default_output_dir.is_empty() {
        return None;
    }
    let dir = Path::new(&cfg.default_output_dir);
    dir.is_dir().then(|| dir.to_path_buf())
}

fn save_request(cfg: &ImageSavingSettings, names: &TemplateContext) -> PickerRequest {
    let (_, filters) = format_filters(cfg.output_format);
    let mut request = PickerRequest::new(PickerKind::SaveFile)
        .titled("Save image")
        .file_name(&suggested_name(cfg, names))
        .starting_in(starting_dir(cfg));
    for (label, extensions) in filters {
        request = request.filter(label, extensions);
    }
    request
}

// ── What is about to be written ──────────────────────────────────────────────

/// `{name}`: the project's name once it has a file, else the first source file's
/// stem, else `stacked`.
fn template_name(project_name: &str, project_path: &str, first_stem: Option<&str>) -> String {
    let name = project_name.trim();
    if !project_path.is_empty() && !name.is_empty() {
        let cut = name.len().saturating_sub(PROJECT_SUFFIX.len());
        if name.len() > PROJECT_SUFFIX.len()
            && name.is_char_boundary(cut)
            && name[cut..].eq_ignore_ascii_case(PROJECT_SUFFIX)
        {
            return name[..cut].to_owned();
        }
        return name.to_owned();
    }
    first_stem.map_or_else(|| "stacked".to_owned(), str::to_owned)
}

/// `{algo}`: the pass that made the viewed result, `source` for a source frame
/// and `stack` for a result that carries no record.
fn algo_label(state: &AppState) -> String {
    let Some(viewed) = state.viewed_path.as_ref() else {
        return "stack".to_owned();
    };
    state
        .result_paths
        .iter()
        .position(|path| path == viewed)
        .map_or_else(
            || "source".to_owned(),
            |index| {
                state
                    .result_meta
                    .get(index)
                    .map(|meta| meta.label.trim())
                    .filter(|label| !label.is_empty())
                    .map_or_else(|| "stack".to_owned(), str::to_owned)
            },
        )
}

/// The dialog's explanation of what Save will write.
fn target_note(target: &SaveTarget, size: Option<(u32, u32)>) -> String {
    let size_text = size.map_or_else(String::new, |(w, h)| format!(" ({w} \u{d7} {h} px)"));
    match target {
        SaveTarget::Result(_) => format!(
            "Saves the newest stacked result at full precision, retouch included{size_text}."
        ),
        SaveTarget::File(src) => format!(
            "Re-encodes the file {}{size_text}. Only the newest result is kept at full \
             precision.",
            feedback::file_label(src)
        ),
    }
}

/// The pixel size of what `target` writes; a file's size is read from its header.
fn target_size(target: &SaveTarget) -> Option<(u32, u32)> {
    match target {
        SaveTarget::Result(session) => Some((
            u32::try_from(session.base.width).ok()?,
            u32::try_from(session.base.height).ok()?,
        )),
        SaveTarget::File(src) => crate::platform::current().image_dimensions(src),
    }
}

/// Where the metadata of a result comes from: the first source frame, and the
/// pre-processing that decides how its orientation tag is fitted.
#[derive(Clone)]
struct ExifSource {
    first: Option<PathBuf>,
    pre: PreprocessingSettings,
    /// The browser build (decided on the UI thread, where the platform lives).
    web: bool,
}

impl ExifSource {
    /// The EXIF block to write (`copy_metadata`). The browser reads the frame
    /// through the platform (its paths are not files); the desktop reads the path
    /// and so works on a worker thread.
    fn block(&self, cfg: &ImageSavingSettings) -> Option<Vec<u8>> {
        if self.web {
            let platform = crate::platform::current();
            let read = |path: &Path| platform.read_bytes(path);
            exif_block(self.first.as_deref(), &self.pre, cfg, Some(&read))
        } else {
            exif_block(self.first.as_deref(), &self.pre, cfg, None)
        }
    }
}

/// Everything one export needs, read from the state at one moment.
struct Snapshot {
    target: SaveTarget,
    names: TemplateContext,
    note: String,
    exif: ExifSource,
}

/// Read the state and the project name; `None` when there is nothing to write.
fn snapshot(ctx: &Ctx, app: &App) -> Option<Snapshot> {
    let project = app.global::<ProjectModel>();
    let (project_name, project_path) = (
        project.get_name().to_string(),
        project.get_path().to_string(),
    );
    let (target, first_stem, frames, algo, first) = {
        let state = ctx.state.borrow();
        let target = state.save_target()?;
        let first_stem = state
            .file_paths
            .first()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned());
        (
            target,
            first_stem,
            state.file_paths.len(),
            algo_label(&state),
            state.file_paths.first().cloned(),
        )
    };
    let exif = ExifSource {
        first,
        pre: ctx.settings.snapshot().preprocessing,
        web: crate::platform::current().is_web(),
    };
    let size = target_size(&target);
    let names = TemplateContext {
        name: template_name(&project_name, &project_path, first_stem.as_deref()),
        algo,
        frames,
        width: size.map_or(0, |(w, _)| w),
        height: size.map_or(0, |(_, h)| h),
    };
    let note = target_note(&target, size);
    Some(Snapshot {
        target,
        names,
        note,
        exif,
    })
}

const NOTHING_TO_SAVE: &str = "Nothing to save: stack or select an image first.";

// ── Entry points ─────────────────────────────────────────────────────────────

/// Open the export dialog (the preview context and the note are refreshed first).
fn open_dialog(ctx: &Ctx, cache: &PreviewContext) {
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    let Some(snapshot) = snapshot(ctx, &app) else {
        feedback::show(&app, MessageKind::Warning, NOTHING_TO_SAVE);
        return;
    };
    *cache.borrow_mut() = snapshot.names;
    let model = app.global::<ExportModel>();
    model.set_target_note(snapshot.note.into());
    model.set_open(true);
}

/// The dialog's Save: into the output folder under the template's name, or, with
/// no folder configured, through a save dialog pre-filled from the template.
fn save_with_template(ctx: &Ctx) {
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    let Some(snapshot) = snapshot(ctx, &app) else {
        feedback::show(&app, MessageKind::Warning, NOTHING_TO_SAVE);
        return;
    };
    let cfg = ctx.settings.snapshot().image_saving;
    let Snapshot {
        target,
        names,
        exif,
        ..
    } = snapshot;
    if let Some(dir) = starting_dir(&cfg) {
        let (ext, _) = format_filters(cfg.output_format);
        let dest = resolve_path(&dir, &cfg.filename_template, &names, ext);
        write_in_background(&app, target, dest, cfg, exif);
        return;
    }
    if !cfg.default_output_dir.is_empty() {
        feedback::show(
            &app,
            MessageKind::Warning,
            &format!(
                "The output folder {} does not exist: choose where to save instead.",
                cfg.default_output_dir
            ),
        );
    }
    ask_for_name(ctx, target, &names, cfg, exif);
}

/// Save As…: always a save dialog, pre-filled from the template.
fn save_as(ctx: &Ctx) {
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    let Some(snapshot) = snapshot(ctx, &app) else {
        feedback::show(&app, MessageKind::Warning, NOTHING_TO_SAVE);
        return;
    };
    let cfg = ctx.settings.snapshot().image_saving;
    ask_for_name(ctx, snapshot.target, &snapshot.names, cfg, snapshot.exif);
}

/// Show the save dialog; encode when a name is chosen.
fn ask_for_name(
    ctx: &Ctx,
    target: SaveTarget,
    names: &TemplateContext,
    cfg: ImageSavingSettings,
    exif: ExifSource,
) {
    let request = save_request(&cfg, names);
    pickers::pick(ctx, request, move |app, _ctx, picked| {
        let Some(dest) = picked.into_iter().next() else {
            return;
        };
        // The native save dialog has already asked before an overwrite.
        write_in_background(app, target, dest, cfg, exif);
    });
}

// ── The worker ───────────────────────────────────────────────────────────────

/// Encode on a worker and report the outcome as a toast. A host without threads
/// (the browser) encodes right here and hands the bytes to the platform, which
/// downloads them.
fn write_in_background(
    app: &App,
    target: SaveTarget,
    dest: PathBuf,
    cfg: ImageSavingSettings,
    exif: ExifSource,
) {
    let platform = crate::platform::current();
    feedback::show_status(app, &format!("Saving {}…", platform.display_path(&dest)));
    if platform.is_web() {
        let (kind, message) = match write_through(platform.as_ref(), &target, &dest, &cfg, &exif) {
            Ok(saved) => (
                MessageKind::Success,
                with_exif_note(success_message(&dest, saved, &target), &dest, &cfg),
            ),
            Err(e) => (
                MessageKind::Error,
                format!("Save failed for {}: {e}", platform.display_path(&dest)),
            ),
        };
        feedback::show(app, kind, &message);
        return;
    }
    let ui = app.as_weak();
    spawn_or_inline("z-stackr-save", move || {
        let (kind, message) = match encode(&target, &dest, &cfg, &exif) {
            Ok(saved) => (
                MessageKind::Success,
                with_exif_note(success_message(&dest, saved, &target), &dest, &cfg),
            ),
            Err(e) => (
                MessageKind::Error,
                format!("Save failed for {}: {e}", dest.display()),
            ),
        };
        feedback::show_from_worker(&ui, kind, message);
    });
}

/// Encode `target` into `dest`.
fn encode(
    target: &SaveTarget,
    dest: &Path,
    cfg: &ImageSavingSettings,
    exif: &ExifSource,
) -> Result<SavedAs, String> {
    match target {
        // The retouched result at `f32`: base and source blended by the stroke mask.
        SaveTarget::Result(session) => {
            save_planar_exif(&session.render_composite(), dest, cfg, exif.block(cfg))
        }
        SaveTarget::File(src) => save_file(src, dest, cfg),
    }
}

/// `message`, with the note that TIFF carries no metadata when that applies.
fn with_exif_note(message: String, dest: &Path, cfg: &ImageSavingSettings) -> String {
    match exif_note(dest, cfg) {
        Some(note) => format!("{message}; {note}"),
        None => message,
    }
}

/// Encode `target` and hand the bytes to `platform` as the file `dest`: the
/// path of a host that cannot write from a worker thread.
fn write_through(
    platform: &dyn Platform,
    target: &SaveTarget,
    dest: &Path,
    cfg: &ImageSavingSettings,
    exif: &ExifSource,
) -> Result<SavedAs, String> {
    let (bytes, saved) = match target {
        SaveTarget::Result(session) => {
            encode_planar_exif(&session.render_composite(), dest, cfg, exif.block(cfg))?
        }
        SaveTarget::File(src) => {
            let source = platform
                .read_bytes(src)
                .map_err(|e| format!("could not open the image to export: {e}"))?;
            encode_image_bytes(&source, dest, cfg)?
        }
    };
    platform
        .write_bytes(dest, &bytes)
        .map_err(|e| e.to_string())?;
    Ok(saved)
}

/// The toast text for a finished save; it says where the pixels came from.
fn success_message(dest: &Path, saved: SavedAs, target: &SaveTarget) -> String {
    let label = saved.label();
    let shown = crate::platform::current().display_path(dest);
    match target {
        SaveTarget::Result(_) => {
            format!("Saved {shown} ({label}, from the full-precision result, retouch included)")
        }
        SaveTarget::File(src) => format!(
            "Saved {shown} ({label}), re-encoded from the file {}: only the newest result is \
             saved at full precision",
            feedback::file_label(src)
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use stacker_algo::hybrid::retouch::RetouchSession;
    use stacker_core::image::PlanarImage;
    use stacker_gui::{
        export::{
            SavedAs,
            filename_template::{TemplateContext, VARIABLES},
        },
        settings::{ImageSavingSettings, OutputFormat},
        state::{AppState, ResultMeta, SaveTarget},
    };

    use super::{
        algo_label, placeholder_help, success_message, suggested_name, target_note, template_name,
    };

    fn names() -> TemplateContext {
        TemplateContext {
            name: "IMG_001".to_owned(),
            algo: "Apex".to_owned(),
            frames: 12,
            width: 100,
            height: 50,
        }
    }

    #[test]
    fn suggested_name_expands_the_template_and_uses_the_format_extension() {
        let cfg = ImageSavingSettings {
            filename_template: "{name}_stacked".to_owned(),
            output_format: OutputFormat::Png,
            ..ImageSavingSettings::default()
        };
        assert_eq!(suggested_name(&cfg, &names()), "IMG_001_stacked.png");
    }

    #[test]
    fn suggested_name_follows_the_whole_template_vocabulary() {
        let cfg = ImageSavingSettings {
            filename_template: "{name}_{algo}_{n}f_{w}x{h}".to_owned(),
            output_format: OutputFormat::Jpeg,
            ..ImageSavingSettings::default()
        };
        assert_eq!(
            suggested_name(&cfg, &names()),
            "IMG_001_Apex_12f_100x50.jpg"
        );
    }

    #[test]
    fn the_placeholder_help_lists_every_variable() {
        let help = placeholder_help();
        for name in VARIABLES {
            assert!(help.contains(&format!("{{{name}}}")), "{help}");
        }
        assert_eq!(
            VARIABLES.len(),
            8,
            "update the manual when a variable is added"
        );
    }

    #[test]
    fn a_saved_project_names_the_export_after_itself() {
        assert_eq!(
            template_name("macro-fly", "C:/p/macro-fly.zstack.toml", Some("IMG_1")),
            "macro-fly"
        );
        assert_eq!(
            template_name("macro-fly.zstack.toml", "C:/p/x", Some("IMG_1")),
            "macro-fly"
        );
        assert_eq!(template_name("MACRO.ZSTACK.TOML", "C:/p/x", None), "MACRO");
    }

    #[test]
    fn an_unsaved_project_falls_back_to_the_first_source_then_to_stacked() {
        assert_eq!(template_name("Untitled", "", Some("IMG_1")), "IMG_1");
        assert_eq!(template_name("Untitled", "", None), "stacked");
        // A project name that is only the suffix is kept whole rather than emptied.
        assert_eq!(
            template_name(".zstack.toml", "C:/p/x", None),
            ".zstack.toml"
        );
    }

    #[test]
    fn the_algo_label_is_read_from_the_viewed_results_record() {
        let mut state = AppState::default();
        state.push_result_with(
            PathBuf::from("out/a.png"),
            ResultMeta {
                label: "Relief".to_owned(),
                ..ResultMeta::default()
            },
        );
        assert_eq!(algo_label(&state), "Relief");

        state.push_result(PathBuf::from("out/b.png"));
        assert_eq!(algo_label(&state), "stack", "no record: a neutral word");

        state.viewed_path = Some(PathBuf::from("src/frame.jpg"));
        assert_eq!(algo_label(&state), "source");

        state.viewed_path = None;
        assert_eq!(algo_label(&state), "stack");
    }

    #[test]
    fn the_note_names_the_source_of_the_pixels_and_the_size() {
        let image = std::sync::Arc::new(PlanarImage::<f32>::new(1, 1));
        let result = SaveTarget::Result(RetouchSession::new(image.clone(), image));
        let note = target_note(&result, Some((6000, 4000)));
        assert!(note.contains("full precision"), "{note}");
        assert!(note.contains("6000 \u{d7} 4000"), "{note}");

        let file = SaveTarget::File(PathBuf::from("dir/frame_01.jpg"));
        let note = target_note(&file, None);
        assert!(note.contains("frame_01.jpg"), "{note}");
        assert!(!note.contains("px"), "{note}");
    }

    #[test]
    fn the_message_says_whether_the_pixels_came_from_the_full_precision_result() {
        let saved = SavedAs {
            format: OutputFormat::Png,
            bit16: true,
        };
        let image = std::sync::Arc::new(PlanarImage::<f32>::new(1, 1));
        let result = SaveTarget::Result(RetouchSession::new(image.clone(), image));
        let text = success_message(Path::new("out.png"), saved, &result);
        assert!(text.contains("16-bit PNG"), "{text}");
        assert!(text.contains("full-precision"), "{text}");

        let file = SaveTarget::File(PathBuf::from("dir/frame_01.jpg"));
        let text = success_message(Path::new("out.png"), saved, &file);
        assert!(
            text.contains("re-encoded from the file frame_01.jpg"),
            "{text}"
        );
    }
}
