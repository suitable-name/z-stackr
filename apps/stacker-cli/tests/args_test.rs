use clap::{CommandFactory, Parser};
use stacker_cli::args::CliArgs;

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

fn parse_with_tile_size(tile_size: &str) -> Result<CliArgs, clap::Error> {
    CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        tile_size,
    ])
}

#[test]
fn tile_size_below_the_minimum_is_rejected_at_parse_time() {
    assert!(parse_with_tile_size("0").is_err(), "0 must be rejected");
    assert!(parse_with_tile_size("63").is_err(), "63 must be rejected");
    assert_eq!(
        parse_with_tile_size("64").expect("64 is valid").tile_size,
        64
    );
}

#[test]
fn write_default_config_requires_config() {
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
        "--write-default-config",
    ]);
    assert!(result.is_err(), "--write-default-config needs --config");
}

#[test]
fn own_output_paths_lists_output_and_log() {
    let args = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "in/out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
        "--log-file",
        "in/run.log",
    ])
    .expect("valid arguments");
    let own = args.own_output_paths();
    // The output, one `out_<algorithm>.png` per algorithm (apex, relief, strata,
    // ai: the names Both / All write) and the log.
    assert_eq!(own.len(), 6);
    assert!(own.iter().any(|p| p.ends_with("out.png")));
    assert!(own.iter().any(|p| p.ends_with("out_strata.png")));
    assert!(own.iter().any(|p| p.ends_with("run.log")));
}

/// `--device` is case-insensitive like `--mode`.
#[test]
fn device_names_are_case_insensitive() {
    use clap::ValueEnum;
    use stacker_cli::args::DeviceRequest;

    for (spelled, expected) in [
        ("cpu", DeviceRequest::Cpu),
        ("CPU", DeviceRequest::Cpu),
        ("Gpu", DeviceRequest::Gpu),
        ("GPU", DeviceRequest::Gpu),
    ] {
        assert_eq!(
            DeviceRequest::from_str(spelled, true),
            Ok(expected),
            "{spelled}"
        );
    }
    assert!(DeviceRequest::from_str("tpu", true).is_err());
}

#[cfg(feature = "nn")]
#[test]
fn device_flag_parses_in_any_case_in_nn_builds() {
    use stacker_cli::args::DeviceRequest;

    let args = CliArgs::try_parse_from([
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "ai",
        "--tile-size",
        "128",
        "--device",
        "GPU",
    ])
    .expect("--device GPU is accepted");
    assert_eq!(args.device, Some(DeviceRequest::Gpu));
}

#[cfg(feature = "nn")]
#[test]
fn align_model_flag_takes_a_model_name_in_nn_builds() {
    let args = parse(&["--align-model", "batchalign_latest"]).expect("--align-model is accepted");
    assert_eq!(args.align_model.as_deref(), Some("batchalign_latest"));
    assert!(parse(&[]).expect("optional").align_model.is_none());
}

/// Without the `nn` feature the AI-only flags are unknown, not silently ignored.
#[cfg(not(feature = "nn"))]
#[test]
fn ai_only_flags_are_rejected_without_the_nn_feature() {
    for flag in ["--model", "--device", "--align-model"] {
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
            flag,
            "x",
        ]);
        assert!(result.is_err(), "{flag} must be rejected in a non-nn build");
    }
}

/// The required options plus `extra`, parsed.
fn parse(extra: &[&str]) -> Result<CliArgs, clap::Error> {
    let mut argv = vec![
        "app",
        "--input-dir",
        "in",
        "--output-file",
        "out.png",
        "--mode",
        "apex",
        "--tile-size",
        "128",
    ];
    argv.extend_from_slice(extra);
    CliArgs::try_parse_from(argv)
}

#[test]
fn optimizer_accepts_auto_lk_nm_in_any_case() {
    use stacker_cli::args::OptimizerArg;
    use stacker_core::settings::OptimizerSetting;

    assert_eq!(parse(&[]).expect("optional").optimizer, None);
    for (spelled, arg, setting) in [
        ("auto", OptimizerArg::Auto, OptimizerSetting::Auto),
        ("lk", OptimizerArg::Lk, OptimizerSetting::LucasKanade),
        ("NM", OptimizerArg::Nm, OptimizerSetting::NelderMead),
        ("Lk", OptimizerArg::Lk, OptimizerSetting::LucasKanade),
    ] {
        let args = parse(&["--optimizer", spelled]).expect(spelled);
        assert_eq!(args.optimizer, Some(arg), "{spelled}");
        assert_eq!(OptimizerSetting::from(arg), setting, "{spelled}");
    }
    assert!(parse(&["--optimizer", "newton"]).is_err());
    assert!(
        parse(&["--optimizer", "lucaskanade"]).is_err(),
        "lk is the spelling"
    );
}

#[test]
fn stacks_accepts_subfolders_and_single_in_any_case() {
    use stacker_cli::args::StacksMode;

    assert_eq!(parse(&[]).expect("optional").stacks, None);
    for (spelled, expected) in [
        ("subfolders", StacksMode::Subfolders),
        ("single", StacksMode::Single),
        ("Single", StacksMode::Single),
    ] {
        let args = parse(&["--stacks", spelled]).expect(spelled);
        assert_eq!(args.stacks, Some(expected), "{spelled}");
    }
    assert!(parse(&["--stacks", "all"]).is_err());
    assert!(parse(&["--stacks"]).is_err(), "the flag needs a value");
}

/// `--no-gpu` is a plain switch, present in the builds that can use a GPU.
#[cfg(any(feature = "gpu", feature = "nn-gpu"))]
#[test]
fn no_gpu_is_a_plain_switch_in_gpu_builds() {
    assert!(!parse(&[]).expect("optional").no_gpu);
    assert!(parse(&["--no-gpu"]).expect("--no-gpu").no_gpu);
    assert!(parse(&["--no-gpu", "extra"]).is_err(), "it takes no value");
}

/// In a build without any GPU backend there is nothing to switch off.
#[cfg(not(any(feature = "gpu", feature = "nn-gpu")))]
#[test]
fn no_gpu_is_unknown_in_builds_without_a_gpu_backend() {
    assert!(parse(&["--no-gpu"]).is_err());
}

#[test]
fn mode_is_a_free_word_checked_when_the_run_starts() {
    // Every algorithm word, old names included, parses; the run validates it.
    for mode in [
        "apex", "Relief", "STRATA", "both", "All", "pmax", "dmap", "bogus",
    ] {
        let args = CliArgs::try_parse_from([
            "app",
            "--input-dir",
            "in",
            "--output-file",
            "out.png",
            "--mode",
            mode,
            "--tile-size",
            "128",
        ])
        .expect(mode);
        assert_eq!(args.mode, mode);
    }
}

#[test]
fn help_lists_the_new_options_and_algorithms() {
    let help = CliArgs::command().render_long_help().to_string();
    for needle in [
        "--optimizer",
        "--stacks",
        "strata",
        "relief",
        "apex",
        "both",
        "all",
    ] {
        assert!(help.contains(needle), "{needle} missing:\n{help}");
    }
}

#[test]
fn required_flags_are_described_in_help() {
    let help = CliArgs::command().render_long_help().to_string();
    let lines: Vec<&str> = help.lines().collect();
    for flag in ["--input-dir", "--output-file", "--mode", "--tile-size"] {
        // The option's own line (the usage line starts with "Usage:").
        let at = lines
            .iter()
            .position(|l| l.trim_start().starts_with(flag))
            .unwrap_or_else(|| panic!("{flag} missing:\n{help}"));
        let described = lines.get(at + 1).is_some_and(|l| !l.trim().is_empty());
        assert!(described, "{flag} has no help text:\n{help}");
    }
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
