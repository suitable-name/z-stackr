use clap::{CommandFactory, Parser};
use stacker_cli::args::{CliArgs, OptimizerArg, StacksMode};

#[test]
fn verify_cli() {
    CliArgs::command().debug_assert();
}

#[test]
fn test_missing_arguments() {
    let result = CliArgs::try_parse_from(["app"]);
    assert!(result.is_err(), "Should fail with missing arguments");
}

#[test]
fn test_invalid_types_paths() {
    let result = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "/invalid/path/due/to/something",
        "--output-file",
        "out.png",
        "--mode",
        "focus",
        "--tile-size",
        "not-a-number",
    ]);
    assert!(
        result.is_err(),
        "Should fail because tile_size is not a number"
    );
}

#[test]
fn test_flag_conflicts_and_unknown() {
    let result = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out",
        "--mode",
        "focus",
        "--tile-size",
        "128",
        "--unknown-conflict-flag",
    ]);
    assert!(result.is_err(), "Should fail with unknown flag");
}

#[test]
fn test_log_file_optional() {
    let args = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
    ])
    .expect("should parse without --log-file");
    assert!(args.log_file.is_none());

    let args = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
        "--log-file",
        "/tmp/stacker.log",
    ])
    .expect("should parse with --log-file");
    assert!(args.log_file.is_some());
}

#[test]
fn test_stacks_flag_defaults_to_none() {
    let args = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
    ])
    .expect("should parse without --stacks");
    assert!(args.stacks.is_none());
}

#[test]
fn test_stacks_flag_round_trip_subfolders_and_single() {
    let args = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out_dir",
        "--mode",
        "apex",
        "--tile-size",
        "128",
        "--stacks",
        "subfolders",
    ])
    .expect("should parse --stacks subfolders");
    assert_eq!(args.stacks, Some(StacksMode::Subfolders));

    let args = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
        "--stacks",
        "single",
    ])
    .expect("should parse --stacks single");
    assert_eq!(args.stacks, Some(StacksMode::Single));
}

#[cfg(feature = "gpu")]
#[test]
fn test_no_gpu_flag_defaults_to_false() {
    let args = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
    ])
    .expect("should parse without --no-gpu");
    assert!(!args.no_gpu);
}

#[cfg(feature = "gpu")]
#[test]
fn test_no_gpu_flag_round_trips() {
    let args = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
        "--no-gpu",
    ])
    .expect("should parse with --no-gpu");
    assert!(args.no_gpu);
}

#[test]
fn test_stacks_flag_rejects_invalid_value() {
    let result = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
        "--stacks",
        "not-a-real-mode",
    ]);
    assert!(
        result.is_err(),
        "Should reject an unrecognised --stacks value"
    );
}

#[test]
fn test_optimizer_flag_defaults_to_none() {
    let args = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
    ])
    .expect("should parse without --optimizer");
    assert!(args.optimizer.is_none());
}

#[test]
fn test_optimizer_flag_round_trip_auto_lk_nm() {
    let args = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
        "--optimizer",
        "auto",
    ])
    .expect("should parse --optimizer auto");
    assert_eq!(args.optimizer, Some(OptimizerArg::Auto));

    let args = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
        "--optimizer",
        "lk",
    ])
    .expect("should parse --optimizer lk");
    assert_eq!(args.optimizer, Some(OptimizerArg::Lk));

    let args = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
        "--optimizer",
        "nm",
    ])
    .expect("should parse --optimizer nm");
    assert_eq!(args.optimizer, Some(OptimizerArg::Nm));
}

#[test]
fn test_optimizer_flag_rejects_invalid_value() {
    let result = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
        "--optimizer",
        "not-a-real-optimizer",
    ]);
    assert!(
        result.is_err(),
        "Should reject an unrecognised --optimizer value"
    );
}
