//! The trainer's command line: strategies, flags, defaults and rejections.
#![cfg(feature = "autodiff")]

use std::path::PathBuf;

use stacker_nn::{
    data::MIN_ALIGN_CROP,
    model::ModelSize,
    train::args::{Args, ArgsError, TrainStrategy, USAGE},
};

fn parse(list: &[&str]) -> Result<Args, ArgsError> {
    let argv: Vec<String> = list.iter().map(|s| (*s).to_owned()).collect();
    Args::parse_from(&argv)
}

#[test]
fn defaults_are_the_pairwise_trainer_of_before() {
    let args = parse(&["--data", "scenes"]).unwrap();
    assert_eq!(args.strategy, TrainStrategy::Pairwise);
    assert_eq!(args.data, PathBuf::from("scenes"));
    assert_eq!(args.out, PathBuf::from("checkpoints"));
    assert_eq!(args.size, ModelSize::M);
    assert_eq!(args.epochs, 40);
    assert!((args.lr - 3e-4).abs() < 1e-12);
    assert_eq!(args.crop, 256);
    assert_eq!(args.seed, 42);
    assert!((args.ss_max - 0.5).abs() < 1e-12);
    assert_eq!(args.ckpt_every, 200);
    assert_eq!(args.cache_mb, 1024);
    assert!(!args.resume && !args.fresh);
    assert_eq!(args.corr_radius, None);
    assert_eq!(args.real_data, None);
    assert!((args.photometric_weight - 0.1).abs() < 1e-12);
    assert_eq!(args.ignored_flags(), Vec::<&str>::new());
}

#[test]
fn every_strategy_name_parses_and_has_its_own_stem() {
    let mut stems = Vec::new();
    for (name, strategy, stem) in [
        ("pairwise", TrainStrategy::Pairwise, "focusmerge"),
        ("batch", TrainStrategy::Batch, "batchmerge"),
        ("align", TrainStrategy::Align, "batchalign"),
        ("fusion-align", TrainStrategy::FusionAlign, "fusionalign"),
    ] {
        let args = parse(&["--data", "d", "--strategy", name]).unwrap();
        assert_eq!(args.strategy, strategy);
        assert_eq!(strategy.as_str(), name);
        assert_eq!(TrainStrategy::parse(name), Some(strategy));
        assert_eq!(strategy.stem_prefix(), stem);
        stems.push(stem);
    }
    stems.sort_unstable();
    stems.dedup();
    assert_eq!(stems.len(), 4, "strategies never share a checkpoint stem");
    assert_eq!(TrainStrategy::ALL.len(), 4);
    assert_eq!(TrainStrategy::parse("nonsense"), None);
    assert!(TrainStrategy::Align.is_alignment() && TrainStrategy::FusionAlign.is_alignment());
    assert!(!TrainStrategy::Batch.is_alignment() && !TrainStrategy::Pairwise.is_alignment());
}

#[test]
fn align_flags_are_read() {
    let args = parse(&[
        "--data",
        "scenes",
        "--strategy",
        "align",
        "--size",
        "xs",
        "--corr-radius",
        "3",
        "--real-data",
        "real",
        "--photometric-weight",
        "0.25",
        "--crop",
        "64",
        "--cache-mb",
        "0",
        "--resume",
        "--fresh",
    ])
    .unwrap();
    assert_eq!(args.strategy, TrainStrategy::Align);
    assert_eq!(args.size, ModelSize::Xs);
    assert_eq!(args.corr_radius, Some(3));
    assert_eq!(args.real_data, Some(PathBuf::from("real")));
    assert!((args.photometric_weight - 0.25).abs() < 1e-12);
    assert_eq!(args.crop, 64);
    assert_eq!(args.cache_mb, 0);
    assert!(args.resume && args.fresh);
    assert!(
        args.ignored_flags().is_empty(),
        "all of them apply to align"
    );
}

#[test]
fn flags_a_strategy_ignores_are_reported() {
    // --corr-radius belongs to the aligners, --real-data and --photometric-weight to align.
    let fusion = parse(&[
        "--data",
        "d",
        "--strategy",
        "fusion-align",
        "--corr-radius",
        "2",
        "--real-data",
        "r",
        "--photometric-weight",
        "0.5",
        "--ss-max",
        "0.2",
    ])
    .unwrap();
    let mut ignored = fusion.ignored_flags();
    ignored.sort_unstable();
    assert_eq!(ignored, ["--photometric-weight", "--real-data", "--ss-max"]);

    let batch = parse(&["--data", "d", "--strategy", "batch", "--corr-radius", "2"]).unwrap();
    assert_eq!(batch.ignored_flags(), ["--corr-radius"]);

    let pairwise = parse(&["--data", "d", "--ss-max", "0.3", "--corr-radius", "5"]).unwrap();
    assert_eq!(pairwise.ignored_flags(), ["--corr-radius"]);
}

#[test]
fn rejections_name_the_flag() {
    assert_eq!(parse(&[]).unwrap_err(), ArgsError::Required("--data <dir>"));
    assert_eq!(
        parse(&["--data"]).unwrap_err(),
        ArgsError::MissingValue("--data".to_owned())
    );
    assert_eq!(
        parse(&["--data", "d", "--bogus", "1"]).unwrap_err(),
        ArgsError::Unknown("--bogus".to_owned())
    );
    assert_eq!(parse(&["-h"]).unwrap_err(), ArgsError::Help);
    assert_eq!(
        parse(&["--data", "d", "--help"]).unwrap_err(),
        ArgsError::Help
    );

    for bad in [
        &["--data", "d", "--strategy", "nope"][..],
        &["--data", "d", "--size", "huge"],
        &["--data", "d", "--epochs", "many"],
        &["--data", "d", "--lr", "0"],
        &["--data", "d", "--lr", "nan"],
        &["--data", "d", "--ss-max", "1.5"],
        &["--data", "d", "--photometric-weight", "-1"],
        &["--data", "d", "--strategy", "align", "--corr-radius", "0"],
    ] {
        assert!(
            matches!(parse(bad), Err(ArgsError::Invalid { .. })),
            "{bad:?} should be rejected"
        );
    }
}

#[test]
fn the_aligners_need_a_crop_the_encoder_can_downsample() {
    let small = (MIN_ALIGN_CROP - 1).to_string();
    for strategy in ["align", "fusion-align"] {
        let err = parse(&["--data", "d", "--strategy", strategy, "--crop", &small]).unwrap_err();
        assert!(matches!(err, ArgsError::Invalid { ref flag, .. } if flag == "--crop"));
        let ok = MIN_ALIGN_CROP.to_string();
        assert!(parse(&["--data", "d", "--strategy", strategy, "--crop", &ok]).is_ok());
    }
    // The fusion strategies accept any crop, as before.
    assert!(parse(&["--data", "d", "--crop", "8"]).is_ok());
}

#[test]
fn ckpt_every_is_at_least_one_and_usage_lists_the_strategies() {
    assert_eq!(
        parse(&["--data", "d", "--ckpt-every", "0"])
            .unwrap()
            .ckpt_every,
        1
    );
    for needle in [
        "--strategy",
        "pairwise|batch|align|fusion-align",
        "--corr-radius",
        "--real-data",
        "--photometric-weight",
        "--resume",
    ] {
        assert!(USAGE.contains(needle), "usage lacks {needle}");
    }
}
