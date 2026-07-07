use std::{
    path::{Path, PathBuf},
    process::Command,
};

/// Prepare an isolated subset directory for one test.
///
/// Each test uses a unique `tag` so the e2e tests never share input/output
/// directories — cargo runs integration tests concurrently, and a shared
/// directory caused one test's cleanup to delete files while the other was
/// mid-run (manifesting as a `NotFound` error when writing the result).
fn prepare_subset_dir(tag: &str) -> (PathBuf, PathBuf) {
    let sample_dir = PathBuf::from("../../sample");
    let test_dir = sample_dir.join(format!("test_subset_{tag}"));
    let result_dir = test_dir.join("result");

    if !sample_dir.exists() {
        return (test_dir, result_dir);
    }

    std::fs::create_dir_all(&result_dir).unwrap();

    // Copy only the first 3 frames and downsize them to make the test faster.
    for i in 0..3 {
        let filename = format!("frame_{i:04}.jpg");
        let src = sample_dir.join(&filename);
        let dst = test_dir.join(&filename);
        if src.exists() {
            let img = image::open(&src).expect("Failed to open source image");
            let resized = img.resize(256, 256, image::imageops::FilterType::Nearest);
            resized.save(&dst).expect("Failed to save resized image");
        }
    }

    (test_dir, result_dir)
}

fn cleanup_subset_dir(test_dir: &Path) {
    if test_dir.exists() {
        let _ = std::fs::remove_dir_all(test_dir);
    }
}

/// Run the CLI end-to-end for one fusion `mode` in its own isolated directory.
fn run_e2e(tag: &str, mode: &str) {
    let (test_dir, result_dir) = prepare_subset_dir(tag);
    if !test_dir.exists() {
        println!("Sample directory not found, skipping e2e test");
        return;
    }

    let output_file = result_dir.join(format!("{mode}_cli_result.jpg"));

    let bin_path = env!("CARGO_BIN_EXE_z-stackr");
    let status = Command::new(bin_path)
        .arg("--input-dir")
        .arg(&test_dir)
        .arg("--output-file")
        .arg(&output_file)
        .arg("--mode")
        .arg(mode)
        .arg("--tile-size")
        .arg("256")
        .status()
        .expect("Failed to execute stacker-cli");

    assert!(status.success(), "stacker-cli failed with status: {status}");
    assert!(output_file.exists(), "Output file was not created");

    cleanup_subset_dir(&test_dir);
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
