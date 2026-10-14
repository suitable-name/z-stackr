//! Batch stacking: an `--input-dir` whose subfolders each hold a stack.
//!
//! # What `--input-dir` can be
//!
//! 1. **Frames only** (no subfolder holds a frame): one stack, one output file
//!    at `--output-file`.
//! 2. **Subfolders with frames, no direct frames**: every direct subfolder that
//!    itself directly holds at least one frame is a candidate stack.
//! 3. **Mixed**: direct frames and frame-holding subfolders. The user says which
//!    is meant with `--stacks`, or is asked on a terminal; nothing is guessed.
//!
//! [`classify_input_dir`] is the pure decision function for this split,
//! [`decide_stacks`] resolves the question and [`plan`] ties them to the command
//! line.
//!
//! # Discovery rule
//!
//! A "subfolder with frames" is a **direct child directory** of `--input-dir`
//! whose own **direct children** include at least one file with a frame
//! extension (`stacker_pipeline::order::has_frame_extension`, the test the
//! directory scan uses, so RAW files count in builds that read them). The search
//! goes one level down only: a subfolder's own subfolders are never inspected. The
//! folder `--output-file` names is never a stack, so a batch can write its results
//! inside the input folder and run again.
//!
//! # Output names (subfolder mode)
//!
//! `--output-file` is the output *directory* (created when missing; a path that
//! has an image extension is refused). Each stack is written to
//! `image_saving.filename_template` with `{name}` replaced by the subfolder name,
//! plus the extension of `image_saving.output_format`. With `--mode both` or `all`
//! every algorithm adds its `_<algorithm>` suffix as in a single run.
//!
//! The stacks run one after the other (each run is parallel inside, and several at
//! once would multiply the memory); a failing stack is reported and the batch
//! goes on. The exit status is an error when any stack failed.

use std::{
    io::IsTerminal,
    path::{Path, PathBuf},
};

use stacker_core::{
    error::StackerError,
    settings::{ImageSavingSettings, StackingSettings},
};
use stacker_pipeline::order::{has_frame_extension, is_excluded};

use crate::{
    args::{CliArgs, StacksMode},
    pipeline::{self, output::templated_file_in},
};

/// The three-way classification of an `--input-dir` ([`classify_input_dir`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputKind {
    /// No subfolder holds frames: the plain "one folder of frames" case.
    ImagesOnly {
        /// The frames directly inside the folder (always more than 0).
        direct_images: usize,
    },
    /// At least one direct subfolder holds frames.
    HasStackSubfolders {
        /// The frame-holding subfolders, sorted by path.
        subfolders: Vec<PathBuf>,
        /// The frames directly inside the folder (0 when it only has subfolders).
        direct_images: usize,
    },
    /// Neither direct frames nor frame-holding subfolders.
    Empty,
}

/// `true` when `path` is an existing file with a frame extension.
fn is_frame_file(path: &Path) -> bool {
    path.is_file() && has_frame_extension(path)
}

/// The frames directly inside `dir`; an unreadable folder counts as having none
/// (the stack run reports the real error when it tries to read it).
fn count_direct_images(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter(|e| is_frame_file(&e.path()))
        .count()
}

/// Classify `input_dir` as described in the module documentation. The folders in
/// `exclude` (the output directory) are never stacks.
///
/// # Errors
/// An I/O error when `input_dir` itself cannot be read.
pub fn classify_input_dir(
    input_dir: &Path,
    exclude: &[PathBuf],
) -> Result<InputKind, StackerError> {
    let direct_images = count_direct_images(input_dir);

    let mut subfolders: Vec<PathBuf> = std::fs::read_dir(input_dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir() && !is_excluded(p, exclude) && count_direct_images(p) > 0)
        .collect();
    subfolders.sort();

    Ok(if !subfolders.is_empty() {
        InputKind::HasStackSubfolders {
            subfolders,
            direct_images,
        }
    } else if direct_images == 0 {
        InputKind::Empty
    } else {
        InputKind::ImagesOnly { direct_images }
    })
}

/// Which target a folder with subfolders means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StacksDecision {
    /// Each subfolder is a stack.
    Subfolders,
    /// Only the frames directly inside the folder.
    Single,
}

impl From<StacksMode> for StacksDecision {
    fn from(mode: StacksMode) -> Self {
        match mode {
            StacksMode::Subfolders => Self::Subfolders,
            StacksMode::Single => Self::Single,
        }
    }
}

/// Resolve [`StacksDecision`] for a folder with `subfolders` frame-holding
/// subfolders and `direct_images` frames of its own.
///
/// The `--stacks` flag always wins. Without it, `ask` is called when
/// `interactive` (stdin is a terminal); otherwise the question cannot be answered
/// and the run stops.
///
/// # Errors
/// [`StackerError::Validation`] when there is neither a flag nor a terminal;
/// whatever `ask` fails with.
pub fn decide_stacks(
    flag: Option<StacksMode>,
    interactive: bool,
    subfolders: usize,
    direct_images: usize,
    ask: impl FnOnce(usize, usize) -> Result<StacksDecision, StackerError>,
) -> Result<StacksDecision, StackerError> {
    if let Some(mode) = flag {
        return Ok(mode.into());
    }
    if interactive {
        return ask(subfolders, direct_images);
    }
    Err(StackerError::Validation(format!(
        "--input-dir holds {subfolders} subfolder(s) with images and {direct_images} image(s) of \
         its own, and stdin is not a terminal: pass `--stacks subfolders` or `--stacks single` \
         to say which one is meant"
    )))
}

/// Ask on the terminal which target is meant, until the answer is understood.
///
/// # Errors
/// [`StackerError::Io`] when stdin cannot be read, [`StackerError::Validation`]
/// when it is closed before an answer.
pub fn prompt_stacks_mode(
    subfolders: usize,
    direct_images: usize,
) -> Result<StacksDecision, StackerError> {
    use std::io::Write;

    eprintln!(
        "'--input-dir' holds {subfolders} subfolder{} with images and {direct_images} image{} \
         directly in the folder.\n\
         Stack each subfolder on its own, or only the images directly in this folder?\n  \
         [s] each subfolder is its own stack\n  \
         [d] only the direct images in this folder\n\
         (pass --stacks subfolders or --stacks single to skip this question)",
        if subfolders == 1 { "" } else { "s" },
        if direct_images == 1 { "" } else { "s" },
    );
    loop {
        eprint!("> ");
        std::io::stderr().flush().ok();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            return Err(StackerError::Validation(
                "stdin closed before the --stacks question was answered".to_owned(),
            ));
        }
        match line.trim().to_ascii_lowercase().as_str() {
            "s" | "subfolders" => return Ok(StacksDecision::Subfolders),
            "d" | "single" => return Ok(StacksDecision::Single),
            _ => eprintln!("Please answer 's' (subfolders) or 'd' (direct images)."),
        }
    }
}

/// `true` when stdin is a terminal: the only case in which the question may be
/// asked.
#[must_use]
pub fn stdin_is_tty() -> bool {
    std::io::stdin().is_terminal()
}

/// What a run does with its `--input-dir`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    /// One stack of the frames directly inside `--input-dir`.
    Single,
    /// One stack per listed subfolder.
    Batch(Vec<PathBuf>),
}

/// `true` when `path` looks like an image file (it has a frame extension), which
/// the batch's output directory must not.
#[must_use]
pub fn looks_like_image_file(path: &Path) -> bool {
    has_frame_extension(path)
}

/// The plan for `args`: classify the input folder and resolve `--stacks`, asking
/// on the terminal when needed.
///
/// # Errors
/// An unreadable input folder, an unanswerable `--stacks` question, or a batch
/// that cannot run: `--monitor` with subfolder mode, or an `--output-file` that
/// looks like an image file.
pub fn plan(args: &CliArgs) -> Result<Plan, StackerError> {
    plan_with(args, stdin_is_tty(), prompt_stacks_mode)
}

/// [`plan`] with the terminal check and the question injected.
///
/// # Errors
/// As [`plan`].
pub fn plan_with(
    args: &CliArgs,
    interactive: bool,
    ask: impl FnOnce(usize, usize) -> Result<StacksDecision, StackerError>,
) -> Result<Plan, StackerError> {
    if args.stacks == Some(StacksMode::Single) {
        return Ok(Plan::Single);
    }
    let kind = classify_input_dir(&args.input_dir, std::slice::from_ref(&args.output_file))?;
    let InputKind::HasStackSubfolders {
        subfolders,
        direct_images,
    } = kind
    else {
        return Ok(Plan::Single);
    };
    let decision = decide_stacks(
        args.stacks,
        interactive,
        subfolders.len(),
        direct_images,
        ask,
    )?;
    if decision == StacksDecision::Single {
        return Ok(Plan::Single);
    }
    if args.monitor {
        return Err(StackerError::Validation(
            "--monitor cannot watch several stacks: run without --monitor, or pass --stacks single \
             to stack only the images directly in --input-dir"
                .to_owned(),
        ));
    }
    if looks_like_image_file(&args.output_file) {
        return Err(StackerError::Validation(format!(
            "with --stacks subfolders, --output-file is the output directory, not an image file \
             (got '{}'); the file name of each stack comes from image_saving.filename_template",
            args.output_file.display()
        )));
    }
    Ok(Plan::Batch(subfolders))
}

/// The file a stack named `name` is written to inside `output_dir`:
/// `filename_template` with `{name}` replaced, plus the extension of
/// `output_format`.
///
/// # Errors
/// [`StackerError::Validation`] when the template produces an empty name.
pub fn resolve_batch_output_path(
    output_dir: &Path,
    name: &str,
    saving: &ImageSavingSettings,
) -> Result<PathBuf, StackerError> {
    templated_file_in(output_dir, saving, name)
}

/// What one subfolder's run came to.
#[derive(Debug)]
pub struct StackOutcome {
    /// The subfolder's name.
    pub name: String,
    /// The files written (one per algorithm), or the error message.
    pub result: Result<Vec<PathBuf>, String>,
}

/// The name of a subfolder: its last path component.
fn folder_name(folder: &Path) -> String {
    folder
        .file_name()
        .map_or_else(|| "stack".to_owned(), |n| n.to_string_lossy().into_owned())
}

/// Stack one subfolder into `output_dir`.
fn run_subfolder(
    args: &CliArgs,
    settings: &StackingSettings,
    folder: &Path,
    name: &str,
    scratch_root: &Path,
) -> Result<Vec<PathBuf>, StackerError> {
    let file = resolve_batch_output_path(&args.output_file, name, &settings.image_saving)?;
    let stack_args = CliArgs {
        input_dir: folder.to_path_buf(),
        output_file: file,
        monitor: false,
        ..args.clone()
    };
    pipeline::run_stack_in(&stack_args, settings, scratch_root)
}

/// Run every folder of `subfolders` as a stack of its own, one after the other.
///
/// The results go into the directory `args.output_file` (created when missing). A
/// failing stack is logged and reported in its outcome; the others still run.
///
/// # Errors
/// [`StackerError::Io`] when the output directory cannot be created.
pub fn run_batch_in(
    args: &CliArgs,
    settings: &StackingSettings,
    subfolders: &[PathBuf],
    scratch_root: &Path,
) -> Result<Vec<StackOutcome>, StackerError> {
    std::fs::create_dir_all(&args.output_file)?;
    let mut outcomes = Vec::with_capacity(subfolders.len());
    for folder in subfolders {
        let name = folder_name(folder);
        println!("=== [{name}] stacking '{}' ===", folder.display());
        tracing::info!(folder = %folder.display(), "batch: starting stack");
        let result = run_subfolder(args, settings, folder, &name, scratch_root).map_err(|error| {
            tracing::error!(folder = %folder.display(), %error, "batch: stack failed");
            error.to_string()
        });
        if let Ok(files) = &result {
            for file in files {
                println!("=== [{name}] done -> {} ===", file.display());
            }
        }
        outcomes.push(StackOutcome { name, result });
    }
    Ok(outcomes)
}

/// [`run_batch_in`] with the system temporary directory as scratch root.
///
/// # Errors
/// As [`run_batch_in`].
pub fn run_batch(
    args: &CliArgs,
    settings: &StackingSettings,
    subfolders: &[PathBuf],
) -> Result<Vec<StackOutcome>, StackerError> {
    run_batch_in(args, settings, subfolders, &std::env::temp_dir())
}

/// Print the summary table of a batch to stdout and return `true` when every
/// stack succeeded.
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
            Ok(files) => {
                let list: Vec<String> = files.iter().map(|f| f.display().to_string()).collect();
                println!("  [ok]     {} -> {}", outcome.name, list.join(", "));
            }
            Err(error) => {
                all_ok = false;
                println!("  [FAILED] {} -> {error}", outcome.name);
            }
        }
    }
    all_ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;
    use stacker_core::settings::{AlignmentModeSetting, OutputFormat};

    fn unique_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "z_stackr_batch_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos())
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create test dir");
        dir
    }

    fn touch(dir: &Path, name: &str) {
        std::fs::write(dir.join(name), b"extension is all that matters").unwrap();
    }

    fn real_frame(dir: &Path, name: &str, seed: u32) {
        RgbImage::from_fn(40, 40, |x, y| {
            let v = x
                .wrapping_mul(31)
                .wrapping_add(y.wrapping_mul(17))
                .wrapping_add(seed * 53);
            let [b0, b1, b2, _] = v.wrapping_mul(0x9E37_79B1).to_le_bytes();
            image::Rgb([b0, b1, b2])
        })
        .save(dir.join(name))
        .unwrap();
    }

    fn args_for(input: &Path, output: &Path) -> CliArgs {
        CliArgs::for_test(input, output, "apex", 64)
    }

    // ── classify_input_dir ───────────────────────────────────────────────

    #[test]
    fn frames_only_folder() {
        let dir = unique_dir("images_only");
        touch(&dir, "a.png");
        touch(&dir, "b.JPG");
        touch(&dir, "readme.txt");
        assert_eq!(
            classify_input_dir(&dir, &[]).unwrap(),
            InputKind::ImagesOnly { direct_images: 2 }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn subfolders_only_folder_lists_them_sorted() {
        let dir = unique_dir("subfolders_only");
        let (b, a) = (dir.join("stack_b"), dir.join("stack_a"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        touch(&a, "1.png");
        touch(&b, "1.tif");
        touch(&b, "2.tif");
        assert_eq!(
            classify_input_dir(&dir, &[]).unwrap(),
            InputKind::HasStackSubfolders {
                subfolders: vec![a, b],
                direct_images: 0
            }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mixed_folder_reports_both_counts() {
        let dir = unique_dir("mixed");
        touch(&dir, "direct1.png");
        touch(&dir, "direct2.png");
        let sub = dir.join("stack_a");
        std::fs::create_dir_all(&sub).unwrap();
        touch(&sub, "1.png");
        assert_eq!(
            classify_input_dir(&dir, &[]).unwrap(),
            InputKind::HasStackSubfolders {
                subfolders: vec![sub],
                direct_images: 2
            }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_folder_and_folders_without_frame_subfolders() {
        let dir = unique_dir("empty");
        assert_eq!(classify_input_dir(&dir, &[]).unwrap(), InputKind::Empty);

        // A subfolder without frames is ignored, and the search goes one level only.
        let notes = dir.join("not_a_stack");
        std::fs::create_dir_all(&notes).unwrap();
        touch(&notes, "notes.txt");
        let deep = dir.join("outer").join("inner");
        std::fs::create_dir_all(&deep).unwrap();
        touch(&deep, "deep.png");
        assert_eq!(classify_input_dir(&dir, &[]).unwrap(), InputKind::Empty);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_output_folder_is_never_a_stack() {
        let dir = unique_dir("outdir");
        let stack = dir.join("stack");
        let results = dir.join("results");
        std::fs::create_dir_all(&stack).unwrap();
        std::fs::create_dir_all(&results).unwrap();
        touch(&stack, "1.png");
        touch(&results, "stack_stacked.png");
        assert_eq!(
            classify_input_dir(&dir, std::slice::from_ref(&results)).unwrap(),
            InputKind::HasStackSubfolders {
                subfolders: vec![stack],
                direct_images: 0
            }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── decide_stacks ────────────────────────────────────────────────────

    #[test]
    fn the_flag_wins_and_nothing_is_asked() {
        let never = |_: usize, _: usize| -> Result<StacksDecision, StackerError> {
            panic!("the flag must make the question unnecessary")
        };
        assert_eq!(
            decide_stacks(Some(StacksMode::Subfolders), true, 2, 1, never).unwrap(),
            StacksDecision::Subfolders
        );
        assert_eq!(
            decide_stacks(Some(StacksMode::Single), false, 2, 1, never).unwrap(),
            StacksDecision::Single
        );
    }

    #[test]
    fn without_a_flag_a_terminal_is_asked_and_a_pipe_is_an_error() {
        let asked = decide_stacks(None, true, 3, 0, |subfolders, direct| {
            assert_eq!((subfolders, direct), (3, 0));
            Ok(StacksDecision::Subfolders)
        })
        .unwrap();
        assert_eq!(asked, StacksDecision::Subfolders);

        let err = decide_stacks(None, false, 3, 0, |_, _| Ok(StacksDecision::Single))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--stacks"), "{err}");
    }

    // ── plan ─────────────────────────────────────────────────────────────

    #[test]
    fn plan_is_single_for_frames_and_for_the_single_flag() {
        let dir = unique_dir("plan_single");
        touch(&dir, "a.png");
        let args = args_for(&dir, &dir.join("out.png"));
        assert_eq!(
            plan_with(&args, false, |_, _| unreachable!()).unwrap(),
            Plan::Single
        );

        let nested = unique_dir("plan_single_nested");
        let sub = nested.join("s");
        std::fs::create_dir_all(&sub).unwrap();
        touch(&sub, "1.png");
        let mut args = args_for(&nested, &nested.join("out"));
        args.stacks = Some(StacksMode::Single);
        assert_eq!(
            plan_with(&args, false, |_, _| unreachable!()).unwrap(),
            Plan::Single
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&nested);
    }

    #[test]
    fn plan_lists_the_subfolders_and_refuses_unusable_batches() {
        let dir = unique_dir("plan_batch");
        let sub = dir.join("s");
        std::fs::create_dir_all(&sub).unwrap();
        touch(&sub, "1.png");

        let mut args = args_for(&dir, &dir.join("out"));
        args.stacks = Some(StacksMode::Subfolders);
        assert_eq!(
            plan_with(&args, false, |_, _| unreachable!()).unwrap(),
            Plan::Batch(vec![sub.clone()])
        );

        // The question is asked when there is no flag and a terminal.
        args.stacks = None;
        assert_eq!(
            plan_with(&args, true, |n, d| {
                assert_eq!((n, d), (1, 0));
                Ok(StacksDecision::Subfolders)
            })
            .unwrap(),
            Plan::Batch(vec![sub])
        );

        // No flag and no terminal: the run stops.
        assert!(plan_with(&args, false, |_, _| unreachable!()).is_err());

        // Subfolder mode and --monitor exclude each other.
        args.stacks = Some(StacksMode::Subfolders);
        args.monitor = true;
        let err = plan_with(&args, false, |_, _| unreachable!())
            .unwrap_err()
            .to_string();
        assert!(err.contains("--monitor"), "{err}");

        // --output-file must be a directory in subfolder mode.
        args.monitor = false;
        args.output_file = dir.join("result.png");
        let err = plan_with(&args, false, |_, _| unreachable!())
            .unwrap_err()
            .to_string();
        assert!(err.contains("output directory"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── output names ─────────────────────────────────────────────────────

    #[test]
    fn output_name_is_the_template_with_the_format_extension() {
        let saving = ImageSavingSettings {
            filename_template: "{name}_stacked".to_owned(),
            output_format: OutputFormat::Tiff,
            ..ImageSavingSettings::default()
        };
        let out = Path::new("batch_out");
        assert_eq!(
            resolve_batch_output_path(out, "coin_03", &saving).unwrap(),
            out.join("coin_03_stacked.tiff")
        );

        let saving = ImageSavingSettings {
            filename_template: "{name}_{name}".to_owned(),
            output_format: OutputFormat::Png,
            ..ImageSavingSettings::default()
        };
        assert_eq!(
            resolve_batch_output_path(out, "x", &saving).unwrap(),
            out.join("x_x.png")
        );

        let saving = ImageSavingSettings {
            filename_template: "  ".to_owned(),
            ..ImageSavingSettings::default()
        };
        assert!(resolve_batch_output_path(out, "x", &saving).is_err());
    }

    #[test]
    fn image_paths_are_not_output_directories() {
        for name in [
            "out.png", "out.PNG", "out.tif", "out.tiff", "out.jpg", "out.jpeg",
        ] {
            assert!(looks_like_image_file(Path::new(name)), "{name}");
        }
        for name in ["out_dir", "batch_results", "out"] {
            assert!(!looks_like_image_file(Path::new(name)), "{name}");
        }
    }

    // ── a batch on a temporary tree ──────────────────────────────────────

    #[test]
    fn a_batch_stacks_every_subfolder_and_reports_the_failing_one() {
        let root = unique_dir("run");
        let input = root.join("input");
        let output = root.join("output");
        let scratch = root.join("scratch");
        for (name, seed) in [("coin_a", 1), ("coin_b", 2)] {
            let folder = input.join(name);
            std::fs::create_dir_all(&folder).unwrap();
            real_frame(&folder, "f0.png", seed);
            real_frame(&folder, "f1.png", seed + 10);
        }
        // A stack whose frames cannot be decoded fails, the others still run.
        let broken = input.join("coin_c");
        std::fs::create_dir_all(&broken).unwrap();
        touch(&broken, "f0.png");
        touch(&broken, "f1.png");
        std::fs::create_dir_all(&scratch).unwrap();

        let mut settings = StackingSettings {
            alignment_mode: AlignmentModeSetting::None,
            auto_cull: false,
            ..StackingSettings::default()
        };
        settings.image_saving.output_format = OutputFormat::Png;
        settings.image_saving.bit_depth = 8;

        let mut args = args_for(&input, &output);
        args.stacks = Some(StacksMode::Subfolders);
        let Plan::Batch(folders) = plan_with(&args, false, |_, _| unreachable!()).unwrap() else {
            panic!("expected a batch");
        };
        assert_eq!(folders.len(), 3);

        let outcomes = run_batch_in(&args, &settings, &folders, &scratch).unwrap();
        let names: Vec<&str> = outcomes.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(names, ["coin_a", "coin_b", "coin_c"]);
        assert!(outcomes[0].result.is_ok() && outcomes[1].result.is_ok());
        assert!(outcomes[2].result.is_err(), "{:?}", outcomes[2]);
        assert!(output.join("coin_a_stacked.png").is_file());
        assert!(output.join("coin_b_stacked.png").is_file());
        assert!(!output.join("coin_c_stacked.png").exists());
        assert!(!print_batch_summary(&outcomes));
        assert_eq!(
            std::fs::read_dir(&scratch).map_or(0, std::iter::Iterator::count),
            0,
            "every stack removes its tile directory"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_second_batch_run_does_not_treat_the_output_folder_as_a_stack() {
        let root = unique_dir("rerun");
        let folder = root.join("coin");
        std::fs::create_dir_all(&folder).unwrap();
        real_frame(&folder, "f0.png", 3);
        real_frame(&folder, "f1.png", 4);
        let output = root.join("results");
        let scratch = unique_dir("rerun_scratch");

        let mut settings = StackingSettings {
            alignment_mode: AlignmentModeSetting::None,
            auto_cull: false,
            ..StackingSettings::default()
        };
        settings.image_saving.output_format = OutputFormat::Png;
        let mut args = args_for(&root, &output);
        args.stacks = Some(StacksMode::Subfolders);

        for _ in 0..2 {
            let Plan::Batch(folders) = plan_with(&args, false, |_, _| unreachable!()).unwrap()
            else {
                panic!("expected a batch");
            };
            assert_eq!(folders, vec![folder.clone()]);
            let outcomes = run_batch_in(&args, &settings, &folders, &scratch).unwrap();
            assert!(print_batch_summary(&outcomes));
        }
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&scratch);
    }
}
