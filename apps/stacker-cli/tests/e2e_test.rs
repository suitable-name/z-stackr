//! End-to-end runs of the `z-stackr` binary on a downsized subset of `sample/`.
//!
//! The tests write only into the system temp directory (one unique directory per
//! test and process, removed on drop even when an assertion fails), never into the
//! repository.  They need the committed `sample/` frames: when the directory is
//! missing the tests fail loudly, unless `Z_STACKR_ALLOW_MISSING_SAMPLE` is set, in
//! which case they print `SKIPPED: ...` to stderr and pass.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

/// Environment variable that turns a missing `sample/` directory into an explicit skip.
const ALLOW_MISSING_SAMPLE: &str = "Z_STACKR_ALLOW_MISSING_SAMPLE";

/// Number of sample frames copied (downsized) into each test's input directory.
const FRAME_COUNT: usize = 3;

/// A scratch directory under the system temp dir, removed when dropped.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("z-stackr-e2e-{tag}-{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).expect("remove stale scratch directory");
        }
        std::fs::create_dir_all(&dir).expect("create scratch directory");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Copy the first [`FRAME_COUNT`] sample frames, downsized, into `input_dir`.
///
/// Returns `false` (after printing an explicit `SKIPPED` line) when `sample/` is
/// absent and the opt-out variable is set; panics when it is absent otherwise.
fn copy_sample_subset(input_dir: &Path) -> bool {
    let sample_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../sample");
    if !sample_dir.is_dir() {
        assert!(
            std::env::var_os(ALLOW_MISSING_SAMPLE).is_some(),
            "sample directory {} not found; the e2e tests need the committed frames \
             (set {ALLOW_MISSING_SAMPLE}=1 to skip them explicitly)",
            sample_dir.display()
        );
        eprintln!(
            "SKIPPED: {} not found and {ALLOW_MISSING_SAMPLE} is set",
            sample_dir.display()
        );
        return false;
    }

    for i in 0..FRAME_COUNT {
        let filename = format!("frame_{i:04}.jpg");
        let src = sample_dir.join(&filename);
        assert!(src.is_file(), "sample frame {} is missing", src.display());
        let img = image::open(&src).expect("Failed to open source image");
        let resized = img.resize(256, 256, image::imageops::FilterType::Nearest);
        resized
            .save(input_dir.join(&filename))
            .expect("Failed to save resized image");
    }
    true
}

/// Run the CLI end-to-end for one fusion `mode` in its own isolated directory,
/// with `extra` arguments, and check that `outputs` (file names inside the result
/// directory) were written and nothing else.
fn run_e2e_expecting(tag: &str, mode: &str, extra: &[&str], outputs: &[&str]) {
    let scratch = ScratchDir::new(tag);
    let input_dir = scratch.path().join("input");
    let result_dir = scratch.path().join("result");
    std::fs::create_dir_all(&input_dir).unwrap();
    std::fs::create_dir_all(&result_dir).unwrap();

    if !copy_sample_subset(&input_dir) {
        return;
    }

    let output_file = result_dir.join(format!("{tag}_cli_result.jpg"));

    let bin_path = env!("CARGO_BIN_EXE_z-stackr");
    let status = Command::new(bin_path)
        .arg("--input-dir")
        .arg(&input_dir)
        .arg("--output-file")
        .arg(&output_file)
        .arg("--mode")
        .arg(mode)
        .arg("--tile-size")
        .arg("256")
        .args(extra)
        .status()
        .expect("Failed to execute stacker-cli");

    assert!(status.success(), "stacker-cli failed with status: {status}");
    let mut found: Vec<String> = std::fs::read_dir(&result_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    found.sort();
    let mut expected: Vec<String> = outputs.iter().map(|s| (*s).to_owned()).collect();
    expected.sort();
    assert_eq!(found, expected, "the files written");
}

/// One algorithm, one output file.
fn run_e2e(tag: &str, mode: &str) {
    let expected = format!("{tag}_cli_result.jpg");
    run_e2e_expecting(tag, mode, &[], &[expected.as_str()]);
}

#[test]
fn test_e2e_apex() {
    run_e2e("apex", "apex");
}

#[test]
fn test_e2e_relief() {
    run_e2e("relief", "relief");
}

#[test]
fn test_e2e_strata() {
    run_e2e("strata", "strata");
}

/// The old mode names still work.
#[test]
fn test_e2e_old_names() {
    run_e2e("pmax", "pmax");
    run_e2e("dmap", "DMap");
}

/// `--optimizer` is accepted and the run succeeds with it.
#[test]
fn test_e2e_optimizer_flag() {
    run_e2e_expecting(
        "optimizer",
        "apex",
        &["--optimizer", "nm"],
        &["optimizer_cli_result.jpg"],
    );
}

/// `both` and `all` write one file per algorithm.
#[test]
fn test_e2e_both_and_all_write_one_file_per_algorithm() {
    run_e2e_expecting(
        "both",
        "both",
        &[],
        &["both_cli_result_relief.jpg", "both_cli_result_apex.jpg"],
    );
    run_e2e_expecting(
        "all",
        "all",
        &[],
        &[
            "all_cli_result_strata.jpg",
            "all_cli_result_relief.jpg",
            "all_cli_result_apex.jpg",
        ],
    );
}

/// `--stacks subfolders` stacks every subfolder into the output directory, and
/// a stdin that is not a terminal needs the flag.
#[test]
fn test_e2e_stacks_subfolders() {
    let scratch = ScratchDir::new("stacks");
    let root = scratch.path().join("input");
    for name in ["coin_a", "coin_b"] {
        let folder = root.join(name);
        std::fs::create_dir_all(&folder).unwrap();
        if !copy_sample_subset(&folder) {
            return;
        }
    }
    let out_dir = scratch.path().join("results");
    let bin_path = env!("CARGO_BIN_EXE_z-stackr");
    let run = |extra: &[&str]| {
        Command::new(bin_path)
            .arg("--input-dir")
            .arg(&root)
            .arg("--output-file")
            .arg(&out_dir)
            .arg("--mode")
            .arg("apex")
            .arg("--tile-size")
            .arg("256")
            .args(extra)
            .stdin(std::process::Stdio::null())
            .status()
            .expect("Failed to execute stacker-cli")
    };

    // The input holds only subfolders; without a terminal and without the flag
    // the question cannot be answered.
    assert!(!run(&[]).success(), "no flag and no terminal must stop");
    assert!(!out_dir.exists() || std::fs::read_dir(&out_dir).unwrap().next().is_none());

    assert!(run(&["--stacks", "subfolders"]).success());
    let mut found: Vec<String> = std::fs::read_dir(&out_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    found.sort();
    assert_eq!(found.len(), 2, "{found:?}");
    assert!(found[0].starts_with("coin_a_stacked."), "{found:?}");
    assert!(found[1].starts_with("coin_b_stacked."), "{found:?}");
}
