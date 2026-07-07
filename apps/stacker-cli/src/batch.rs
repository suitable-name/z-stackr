//! Batch-processing support for `--input-dir` folders that contain
//! image-bearing subfolders.
//!
//! # The three-way split
//!
//! `--input-dir` can point at one of three shapes of folder:
//!
//! 1. **Images only** (no subfolder contains images) — the plain case: one
//!    stack, one output file at `--output-file`.
//! 2. **Subfolders with images, no direct images** — every direct child
//!    directory that itself directly contains at least one recognised image
//!    is a candidate stack.
//! 3. **Mixed** — both direct images *and* image-bearing subfolders are
//!    present. The user is asked (interactively) or must say (via
//!    `--stacks`) which of the two they mean; there is no folder shape that
//!    silently guesses.
//!
//! [`classify_input_dir`] is the pure, unit-tested decision function for
//! this split. The interactive prompt itself ([`prompt_stacks_mode`]) is a
//! thin, untested wrapper isolated specifically so the decision logic above
//! it stays testable without a real terminal.
//!
//! # Discovery rule
//!
//! A "subfolder with images" is a **direct child directory** of
//! `--input-dir` whose own **direct children** include at least one file
//! with a recognised image extension (standard formats always; RAW
//! extensions when built with the `raw` feature) — exactly the same
//! extension test [`stacker_pipeline::collect_image_paths`] uses, so batch
//! discovery automatically tracks whatever formats that function recognises
//! (including RAW, without this module hardcoding a second extension list).
//! This is **non-recursive beyond one level**: a subfolder's own
//! subfolders are never inspected, and a subfolder with no direct images
//! (even if *its* subfolders have some) is not counted as a stack and is
//! silently ignored.
//!
//! # Output naming (subfolder-batch mode only)
//!
//! In subfolder-batch mode `--output-file` is reinterpreted as an output
//! *directory* (created if missing). Each subfolder's output filename reuses
//! the exact naming rule the GUI's Save dialog uses
//! (`apps/stacker-gui/src/main.rs`'s `perform_save_current_image`): the
//! configured `image_saving.filename_template` with `{name}` substituted by
//! the subfolder's name, plus the extension for the configured
//! `image_saving.output_format` (via
//! [`stacker_core::settings::OutputFormat::extension`], the same helper the
//! GUI mapping was factored out into). Single-stack mode is unaffected:
//! `--output-file` keeps meaning an explicit output file.

use std::{
    io::IsTerminal,
    path::{Path, PathBuf},
};

use stacker_core::{error::StackerError, settings::ImageSavingSettings};

/// The three-way classification of an `--input-dir` folder.
///
/// Pure function output ([`classify_input_dir`]) — no I/O side effects, no
/// prompting; every field is plain data so this is trivially unit-testable
/// against temp-dir fixtures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputKind {
    /// No image-bearing subfolders: the plain "one folder of images" case.
    /// Carries the direct image count purely for diagnostics/logging (it is
    /// always `> 0`, since an all-empty folder is [`InputKind::Empty`]
    /// instead).
    ImagesOnly { direct_images: usize },
    /// At least one direct child subfolder contains images. `direct_images`
    /// is the count of images directly inside `--input-dir` itself (`0` in
    /// the pure subfolder case, `> 0` in the mixed case) and `subfolders` is
    /// the sorted list of image-bearing subfolder paths.
    HasStackSubfolders {
        subfolders: Vec<PathBuf>,
        direct_images: usize,
    },
    /// Neither direct images nor image-bearing subfolders were found.
    Empty,
}

/// Returns `true` if `path`'s extension is recognised as an image format —
/// the exact same rule `stacker_pipeline::collect_image_paths` filters
/// directory scans by (standard formats always; RAW extensions only when
/// this binary was built with the `raw` feature). Extension-only (no
/// `is_file()` check), so it can also be used on paths that don't exist yet
/// (see [`looks_like_image_file`]).
fn has_recognised_image_extension(path: &Path) -> bool {
    let Some(ext) = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
    else {
        return false;
    };
    let is_standard = matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "tif" | "tiff");
    let is_raw = cfg!(feature = "raw") && stacker_core::io::is_raw_extension(&ext);
    is_standard || is_raw
}

/// Returns `true` if `path` is an existing file with a recognised image
/// extension — see [`has_recognised_image_extension`] for the extension
/// rule. Used by the subfolder/direct-image scan in
/// [`classify_input_dir`], which only ever looks at paths that already
/// exist (unlike [`looks_like_image_file`], which must also work for a
/// not-yet-created output path).
fn is_recognised_image_file(path: &Path) -> bool {
    path.is_file() && has_recognised_image_extension(path)
}

/// Count direct-child image files inside `dir` (non-recursive). Returns `0`
/// if `dir` cannot be read (treated as "no images" rather than propagating
/// the error — callers that need a hard I/O error already get one from
/// `collect_image_paths` when they actually try to stack the folder).
fn count_direct_images(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| is_recognised_image_file(p))
        .count()
}

/// Classify `input_dir` per the module docs' three-way split.
///
/// Non-recursive beyond one level: only `input_dir`'s direct children are
/// inspected for direct images, and only each direct child *directory*'s own
/// direct children are inspected when deciding whether that subfolder
/// "contains images" — a subfolder's own subfolders are never examined.
///
/// # Errors
/// Returns [`StackerError`] only if `input_dir` itself cannot be read.
pub fn classify_input_dir(input_dir: &Path) -> Result<InputKind, StackerError> {
    let direct_images = count_direct_images(input_dir);

    let mut subfolders: Vec<PathBuf> = std::fs::read_dir(input_dir)?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir() && count_direct_images(p) > 0)
        .collect();
    subfolders.sort();

    if subfolders.is_empty() {
        if direct_images == 0 {
            Ok(InputKind::Empty)
        } else {
            Ok(InputKind::ImagesOnly { direct_images })
        }
    } else {
        Ok(InputKind::HasStackSubfolders {
            subfolders,
            direct_images,
        })
    }
}

/// Non-interactive / interactive resolution of "which target did the user
/// mean" for an `--input-dir` that [`classify_input_dir`] reported as
/// [`InputKind::HasStackSubfolders`].
///
/// `flag` is `--stacks`'s value, if given — it always wins and skips the
/// prompt (even on a TTY). Otherwise, if stdin is a TTY, the caller should
/// use [`prompt_stacks_mode`]; if not, this is a hard error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StacksDecision {
    Subfolders,
    Single,
}

/// Resolve [`StacksDecision`] from an explicit `--stacks` flag value.
///
/// Does not touch stdin at all. Returns `None` when no flag was given — the
/// caller then decides between prompting (TTY) or erroring (no TTY).
#[must_use]
pub const fn decision_from_flag(flag: Option<crate::args::StacksMode>) -> Option<StacksDecision> {
    match flag {
        Some(crate::args::StacksMode::Subfolders) => Some(StacksDecision::Subfolders),
        Some(crate::args::StacksMode::Single) => Some(StacksDecision::Single),
        None => None,
    }
}

/// Thin, deliberately untested wrapper around an interactive stdin/stderr prompt.
///
/// Kept to the absolute minimum of logic (read a line, match it)
/// specifically so every decision this module makes is otherwise covered by
/// unit tests against [`classify_input_dir`] / [`decision_from_flag`] /
/// [`resolve_batch_output_path`] — only this function actually touches a
/// real terminal, and it is only ever called after
/// [`std::io::IsTerminal::is_terminal`] has confirmed stdin is interactive.
///
/// Prints the prompt (with the subfolder/direct-image counts, per the
/// product requirement) to stderr, reads one line from stdin, and loops
/// until the answer is recognised.
///
/// # Errors
/// Returns [`StackerError`] only if stdin is closed/unreadable before a
/// valid answer is given.
pub fn prompt_stacks_mode(
    subfolders: usize,
    direct_images: usize,
) -> Result<StacksDecision, StackerError> {
    use std::io::Write;

    eprintln!(
        "'--input-dir' contains {subfolders} subfolder{s1} with images and {direct_images} image{s2} directly \
         in the folder.\n\
         Stack each subfolder independently, or only the images directly in this folder?\n  \
         [s] each subfolder is its own stack\n  \
         [d] only the direct images in this folder\n\
         (pass --stacks subfolders / --stacks single to skip this prompt next time)",
        s1 = if subfolders == 1 { "" } else { "s" },
        s2 = if direct_images == 1 { "" } else { "s" },
    );

    loop {
        eprint!("> ");
        std::io::stderr().flush().ok();
        let mut line = String::new();
        let n = std::io::stdin()
            .read_line(&mut line)
            .map_err(StackerError::Io)?;
        if n == 0 {
            return Err(StackerError::AlignmentFailed(
                "stdin closed before answering the --stacks prompt".to_owned(),
            ));
        }
        match line.trim().to_ascii_lowercase().as_str() {
            "s" | "subfolders" => return Ok(StacksDecision::Subfolders),
            "d" | "single" => return Ok(StacksDecision::Single),
            _ => eprintln!("Please answer 's' (subfolders) or 'd' (direct images)."),
        }
    }
}

/// Returns `true` if stdin is an interactive terminal.
///
/// The sole gate for whether [`prompt_stacks_mode`] may be used at all.
/// Exists as its own function purely so call sites read as intent ("are we
/// allowed to prompt?") rather than an inline `std::io::stdin().is_terminal()`.
#[must_use]
pub fn stdin_is_tty() -> bool {
    std::io::stdin().is_terminal()
}

/// Resolve the per-subfolder output file path in subfolder-batch mode.
///
/// `output_dir` is `--output-file` reinterpreted as a directory (the caller
/// is responsible for creating it — see [`ensure_batch_output_dir`] — and
/// for having already rejected an `--output-file` that looks like an image
/// file via [`looks_like_image_file`]). `subfolder_name`
/// is the subfolder's own directory name (e.g. `"stack_03"` for
/// `.../input/stack_03`). `image_saving` supplies the filename template and
/// output format exactly as the GUI's Save dialog would.
///
/// Naming rule (mirrors `apps/stacker-gui/src/main.rs`'s
/// `perform_save_current_image`): `filename_template` with every `{name}`
/// occurrence replaced by `subfolder_name`, plus a `.` and
/// `image_saving.output_format.extension()`.
///
/// # Example
/// `filename_template = "{name}_stacked"`, `output_format = Tiff`,
/// `subfolder_name = "coin_03"` → `coin_03_stacked.tiff`.
#[must_use]
pub fn resolve_batch_output_path(
    output_dir: &Path,
    subfolder_name: &str,
    image_saving: &ImageSavingSettings,
) -> PathBuf {
    let stem = image_saving
        .filename_template
        .replace("{name}", subfolder_name);
    let filename = format!("{stem}.{ext}", ext = image_saving.output_format.extension());
    output_dir.join(filename)
}

/// Returns `true` if `path` has an extension recognised as an image format.
///
/// Standard formats always, or RAW in `raw`-feature builds — used to reject
/// an `--output-file` that looks like a single file when subfolder-batch
/// mode needs it to be a directory instead. Deliberately does not check
/// whether `path` exists: the whole point is to catch the mistake *before*
/// creating a directory there.
#[must_use]
pub fn looks_like_image_file(path: &Path) -> bool {
    has_recognised_image_extension(path)
}

/// Create `output_dir` (and any missing parents) if it doesn't already exist.
///
/// Separated out from [`resolve_batch_output_path`] so the pure path
/// computation stays side-effect-free and unit-testable without touching
/// the filesystem.
///
/// # Errors
/// Returns [`StackerError`] if directory creation fails.
pub fn ensure_batch_output_dir(output_dir: &Path) -> Result<(), StackerError> {
    std::fs::create_dir_all(output_dir)?;
    Ok(())
}

/// One subfolder's batch-run outcome, for the end-of-run summary table.
#[derive(Debug)]
pub struct StackOutcome {
    pub name: String,
    pub result: Result<PathBuf, String>,
}

/// Render the end-of-batch summary table to stdout (ok/failed per
/// subfolder, in the order they were run) and return `true` if every stack
/// succeeded.
///
/// A failing subfolder never aborts the batch — the caller logs the error
/// and continues to the next subfolder; this function is only responsible
/// for the final report and the aggregate exit-code decision.
#[must_use]
pub fn print_batch_summary(outcomes: &[StackOutcome]) -> bool {
    println!();
    println!(
        "Batch summary ({} stack{}):",
        outcomes.len(),
        if outcomes.len() == 1 { "" } else { "s" }
    );
    let mut all_ok = true;
    for outcome in outcomes {
        match &outcome.result {
            Ok(path) => println!("  [ok]     {} -> {}", outcome.name, path.display()),
            Err(e) => {
                all_ok = false;
                println!("  [FAILED] {} -> {e}", outcome.name);
            }
        }
    }
    all_ok
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "z_stackr_batch_test_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos())
        ));
        std::fs::create_dir_all(&dir).expect("create unique temp test dir");
        dir
    }

    fn touch_image(dir: &Path, filename: &str) {
        std::fs::write(
            dir.join(filename),
            b"not a real image, extension is all that matters",
        )
        .unwrap();
    }

    fn touch_non_image(dir: &Path, filename: &str) {
        std::fs::write(dir.join(filename), b"irrelevant").unwrap();
    }

    // ── classify_input_dir ───────────────────────────────────────────────

    #[test]
    fn classify_images_only_folder() {
        let dir = unique_temp_dir("images_only");
        touch_image(&dir, "a.png");
        touch_image(&dir, "b.jpg");
        touch_non_image(&dir, "readme.txt");

        let kind = classify_input_dir(&dir).unwrap();
        assert_eq!(kind, InputKind::ImagesOnly { direct_images: 2 });

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn classify_subfolders_only_folder() {
        let dir = unique_temp_dir("subfolders_only");
        let sub_a = dir.join("stack_a");
        let sub_b = dir.join("stack_b");
        std::fs::create_dir_all(&sub_a).unwrap();
        std::fs::create_dir_all(&sub_b).unwrap();
        touch_image(&sub_a, "1.png");
        touch_image(&sub_b, "1.tif");
        touch_image(&sub_b, "2.tif");

        let kind = classify_input_dir(&dir).unwrap();
        match kind {
            InputKind::HasStackSubfolders {
                subfolders,
                direct_images,
            } => {
                assert_eq!(direct_images, 0);
                assert_eq!(subfolders, vec![sub_a, sub_b]);
            }
            other => panic!("expected HasStackSubfolders, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn classify_mixed_folder_reports_both_counts() {
        let dir = unique_temp_dir("mixed");
        touch_image(&dir, "direct1.png");
        touch_image(&dir, "direct2.png");
        let sub = dir.join("stack_a");
        std::fs::create_dir_all(&sub).unwrap();
        touch_image(&sub, "1.png");

        let kind = classify_input_dir(&dir).unwrap();
        match kind {
            InputKind::HasStackSubfolders {
                subfolders,
                direct_images,
            } => {
                assert_eq!(direct_images, 2);
                assert_eq!(subfolders, vec![sub]);
            }
            other => panic!("expected HasStackSubfolders, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn classify_empty_folder() {
        let dir = unique_temp_dir("empty");
        let kind = classify_input_dir(&dir).unwrap();
        assert_eq!(kind, InputKind::Empty);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn classify_ignores_subfolder_without_images() {
        let dir = unique_temp_dir("sub_without_images");
        let sub_empty = dir.join("not_a_stack");
        std::fs::create_dir_all(&sub_empty).unwrap();
        touch_non_image(&sub_empty, "notes.txt");
        let sub_real = dir.join("real_stack");
        std::fs::create_dir_all(&sub_real).unwrap();
        touch_image(&sub_real, "1.png");

        let kind = classify_input_dir(&dir).unwrap();
        match kind {
            InputKind::HasStackSubfolders {
                subfolders,
                direct_images,
            } => {
                assert_eq!(direct_images, 0);
                // Only the subfolder that actually has images is counted —
                // `not_a_stack` (images-less) is silently ignored.
                assert_eq!(subfolders, vec![sub_real]);
            }
            other => panic!("expected HasStackSubfolders, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn classify_ignores_grandchildren_non_recursive() {
        // A subfolder whose own direct children have no images, but whose
        // grandchild directory does, must NOT be counted — discovery is
        // non-recursive beyond one level.
        let dir = unique_temp_dir("grandchildren");
        let sub = dir.join("outer");
        let grandchild = sub.join("inner");
        std::fs::create_dir_all(&grandchild).unwrap();
        touch_image(&grandchild, "deep.png");

        let kind = classify_input_dir(&dir).unwrap();
        assert_eq!(kind, InputKind::Empty);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── decision_from_flag ───────────────────────────────────────────────

    #[test]
    fn decision_from_flag_maps_both_variants() {
        assert_eq!(
            decision_from_flag(Some(crate::args::StacksMode::Subfolders)),
            Some(StacksDecision::Subfolders)
        );
        assert_eq!(
            decision_from_flag(Some(crate::args::StacksMode::Single)),
            Some(StacksDecision::Single)
        );
        assert_eq!(decision_from_flag(None), None);
    }

    // ── resolve_batch_output_path ────────────────────────────────────────

    #[test]
    fn resolve_batch_output_path_substitutes_name_and_extension() {
        let saving = ImageSavingSettings {
            filename_template: "{name}_stacked".to_owned(),
            output_format: stacker_core::settings::OutputFormat::Tiff,
            ..ImageSavingSettings::default()
        };

        let out_dir = Path::new("/tmp/batch_out");
        let path = resolve_batch_output_path(out_dir, "coin_03", &saving);
        assert_eq!(path, out_dir.join("coin_03_stacked.tiff"));
    }

    #[test]
    fn resolve_batch_output_path_honours_configured_format_extension() {
        let saving = ImageSavingSettings {
            filename_template: "{name}".to_owned(),
            output_format: stacker_core::settings::OutputFormat::Jpeg,
            ..ImageSavingSettings::default()
        };

        let out_dir = Path::new("/tmp/batch_out");
        let path = resolve_batch_output_path(out_dir, "leaf", &saving);
        assert_eq!(path, out_dir.join("leaf.jpg"));
    }

    #[test]
    fn resolve_batch_output_path_handles_multiple_name_occurrences() {
        let saving = ImageSavingSettings {
            filename_template: "{name}_{name}".to_owned(),
            output_format: stacker_core::settings::OutputFormat::Png,
            ..ImageSavingSettings::default()
        };

        let out_dir = Path::new("/tmp/batch_out");
        let path = resolve_batch_output_path(out_dir, "x", &saving);
        assert_eq!(path, out_dir.join("x_x.png"));
    }

    // ── looks_like_image_file (the batch-mode --output-file guard) ───────

    #[test]
    fn looks_like_image_file_recognises_standard_extensions() {
        for name in [
            "out.png", "out.PNG", "out.tif", "out.tiff", "out.jpg", "out.jpeg",
        ] {
            assert!(
                looks_like_image_file(Path::new(name)),
                "expected '{name}' to look like an image file"
            );
        }
    }

    #[test]
    fn looks_like_image_file_rejects_directory_like_paths() {
        for name in ["out_dir", "batch_results", "out.", "out"] {
            assert!(
                !looks_like_image_file(Path::new(name)),
                "expected '{name}' to NOT look like an image file"
            );
        }
    }
}
