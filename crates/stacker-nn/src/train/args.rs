//! The command line of `z-stackr-train`, parsed in the library so it can be tested.
//!
//! [`TrainStrategy`] names the four things the trainer can train and decides the
//! checkpoint file stems; [`Args::parse_from`] turns an argument list into an
//! [`Args`] or a precise [`ArgsError`]. The binary only prints the error and the
//! [`USAGE`] text.

use std::{path::PathBuf, str::FromStr};

use crate::{data::MIN_ALIGN_CROP, model::ModelSize};

/// What `--strategy` selects.
///
/// | strategy | network | training data | checkpoint stem |
/// |---|---|---|---|
/// | `pairwise` | `FocusMergeNet` | `02_blender_focus_stack.py` scenes | `focusmerge` |
/// | `batch` | `BatchMergeNet` | the same scenes | `batchmerge` |
/// | `align` | `BatchAlignNet` | `03_simulate_misalignment.py` scenes | `batchalign` |
/// | `fusion-align` | `FusionAlignNet` | the same scenes | `fusionalign` |
///
/// Each strategy writes `<stem>_latest.model.mpk`, `<stem>_latest.optim.mpk`,
/// `<stem>_epoch_NNN.mpk` and `<stem>_final.mpk` (plus manifests) into `--out`, so
/// strategies sharing an output directory never overwrite one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrainStrategy {
    /// Recurrent pairwise merge (the default, and the only strategy before 1.1).
    Pairwise,
    /// Whole-stack merge in one forward pass.
    Batch,
    /// Whole-stack affine alignment.
    Align,
    /// Streaming alignment of one frame against the reference.
    FusionAlign,
}

impl TrainStrategy {
    /// Every strategy, in `--help` order.
    pub const ALL: [Self; 4] = [Self::Pairwise, Self::Batch, Self::Align, Self::FusionAlign];

    /// Parse a `--strategy` value (`pairwise`, `batch`, `align`, `fusion-align`).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|v| v.as_str() == s)
    }

    /// The `--strategy` spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pairwise => "pairwise",
            Self::Batch => "batch",
            Self::Align => "align",
            Self::FusionAlign => "fusion-align",
        }
    }

    /// The checkpoint file stem prefix (`focusmerge`, `batchmerge`, `batchalign`,
    /// `fusionalign`).
    #[must_use]
    pub const fn stem_prefix(self) -> &'static str {
        match self {
            Self::Pairwise => "focusmerge",
            Self::Batch => "batchmerge",
            Self::Align => "batchalign",
            Self::FusionAlign => "fusionalign",
        }
    }

    /// `true` for the two alignment strategies, which read
    /// `03_simulate_misalignment.py` scenes.
    #[must_use]
    pub const fn is_alignment(self) -> bool {
        matches!(self, Self::Align | Self::FusionAlign)
    }
}

/// The help text of `z-stackr-train`.
pub const USAGE: &str = "\
z-stackr-train --data <scenes_dir> [--out <dir>] [--size xs|s|m|l|xl|xxl]
               [--strategy pairwise|batch|align|fusion-align] [--epochs N] [--lr F]
               [--crop N] [--seed N] [--ss-max F] [--ckpt-every N] [--cache-mb N]
               [--corr-radius N] [--real-data <dir>] [--photometric-weight F]
               [--resume [--fresh]]

--data        directory of scene sub-directories, each with a metadata.json:
              pairwise, batch   02_blender_focus_stack.py output (frames, masks, all-focus)
              align, fusion-align   03_simulate_misalignment.py output (alignment_gt)
--strategy    what to train (default pairwise). Each strategy has its own checkpoint
              stems (focusmerge, batchmerge, batchalign, fusionalign) in --out.
--seed N      seeds weight initialisation, scene order, crops and scheduled sampling
              (default 42); GPU kernels may still be non-deterministic
--ss-max F    scheduled sampling, pairwise only (default 0.5)
--cache-mb N  budget for decoded scenes kept between steps (default 1024, 0 = none)
--corr-radius N       align, fusion-align: correlation search radius (default: the
              preset's, 4); recorded in the manifest
--real-data DIR       align only: directory of plain image-folder stacks (no
              metadata); the last 25% of epochs add the photometric fine-tune loss
--photometric-weight F  align only: weight of that term (default 0.1)
--resume      continue from the rolling checkpoint of the same strategy in --out;
              add --fresh to start from scratch when there is none";

/// Why an argument list was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArgsError {
    /// `-h` / `--help` was given; the caller prints [`USAGE`] and exits 0.
    #[error("help requested")]
    Help,
    /// A flag the trainer does not know.
    #[error("unknown argument: {0}")]
    Unknown(String),
    /// A flag that takes a value was last on the line.
    #[error("missing value for {0}")]
    MissingValue(String),
    /// A flag's value could not be parsed or is out of range.
    #[error("invalid value '{value}' for {flag}: {expected}")]
    Invalid {
        /// The flag.
        flag: String,
        /// The offending text.
        value: String,
        /// What was expected.
        expected: String,
    },
    /// A required flag is absent.
    #[error("{0} is required")]
    Required(&'static str),
}

/// Every flag that takes a value.
const VALUE_FLAGS: [&str; 13] = [
    "--data",
    "--out",
    "--size",
    "--strategy",
    "--epochs",
    "--lr",
    "--crop",
    "--seed",
    "--ss-max",
    "--ckpt-every",
    "--cache-mb",
    "--corr-radius",
    "--real-data",
];

/// Parsed command-line arguments of `z-stackr-train`.
#[derive(Debug, Clone)]
pub struct Args {
    /// `--data`: directory of scene sub-directories.
    pub data: PathBuf,
    /// `--out`: checkpoint directory (default `checkpoints`).
    pub out: PathBuf,
    /// `--size`: capacity preset (default `m`).
    pub size: ModelSize,
    /// `--strategy` (default pairwise).
    pub strategy: TrainStrategy,
    /// `--epochs` (default 40).
    pub epochs: usize,
    /// `--lr`: peak learning rate (default 3e-4).
    pub lr: f64,
    /// `--crop`: training crop side (default 256).
    pub crop: usize,
    /// `--seed` (default 42).
    pub seed: u64,
    /// `--ss-max`: peak scheduled-sampling probability, pairwise only (default 0.5).
    pub ss_max: f64,
    /// `--ckpt-every`: rolling checkpoint every N optimiser steps (default 200).
    pub ckpt_every: usize,
    /// `--resume`.
    pub resume: bool,
    /// `--fresh` (with `--resume`): start from scratch when nothing is usable.
    pub fresh: bool,
    /// `--cache-mb`: decoded-scene budget in MiB (default 1024, 0 = none).
    pub cache_mb: usize,
    /// `--corr-radius`: correlation radius of the alignment networks.
    pub corr_radius: Option<usize>,
    /// `--real-data`: directory of unlabelled real stacks (align only).
    pub real_data: Option<PathBuf>,
    /// `--photometric-weight`: weight of the fine-tune term (default 0.1).
    pub photometric_weight: f64,
    /// The value flags that were given on the command line.
    given: Vec<&'static str>,
}

/// Parse `value` as `T`, naming `flag` and `expected` on failure.
fn parse_value<T: FromStr>(flag: &str, value: &str, expected: &str) -> Result<T, ArgsError> {
    value.parse().map_err(|_| ArgsError::Invalid {
        flag: flag.to_owned(),
        value: value.to_owned(),
        expected: expected.to_owned(),
    })
}

/// A finite `f64` in `lo..=hi`; `expected` describes the range for the error.
fn parse_f64_in(
    flag: &str,
    value: &str,
    (lo, hi): (f64, f64),
    expected: &str,
) -> Result<f64, ArgsError> {
    let v: f64 = parse_value(flag, value, expected)?;
    if v.is_finite() && (lo..=hi).contains(&v) {
        Ok(v)
    } else {
        Err(ArgsError::Invalid {
            flag: flag.to_owned(),
            value: value.to_owned(),
            expected: expected.to_owned(),
        })
    }
}

impl Args {
    /// Parse an argument list (without the program name).
    ///
    /// # Errors
    /// [`ArgsError::Help`] for `-h` / `--help`; [`ArgsError::Unknown`],
    /// [`ArgsError::MissingValue`] or [`ArgsError::Invalid`] for a bad flag
    /// (`--lr` must be positive and finite, `--ss-max` in `0..=1`,
    /// `--photometric-weight` finite and non-negative, `--corr-radius` at least 1,
    /// `--crop` at least [`MIN_ALIGN_CROP`] for the alignment strategies);
    /// [`ArgsError::Required`] when `--data` is absent.
    #[expect(
        clippy::too_many_lines,
        reason = "one flat table of flags, each arm a line or two"
    )]
    #[expect(
        clippy::similar_names,
        reason = "`argv` is the raw list and `args` the parsed result; the pair is conventional"
    )]
    pub fn parse_from(argv: &[String]) -> Result<Self, ArgsError> {
        let mut args = Self {
            data: PathBuf::new(),
            out: PathBuf::from("checkpoints"),
            size: ModelSize::M,
            strategy: TrainStrategy::Pairwise,
            epochs: 40,
            lr: 3e-4,
            crop: 256,
            seed: 42,
            ss_max: 0.5,
            ckpt_every: 200,
            resume: false,
            fresh: false,
            cache_mb: 1024,
            corr_radius: None,
            real_data: None,
            photometric_weight: 0.1,
            given: Vec::new(),
        };
        let mut have_data = false;
        let mut photometric_given = false;

        let mut i = 0;
        while i < argv.len() {
            let flag = argv[i].as_str();
            match flag {
                "-h" | "--help" => return Err(ArgsError::Help),
                "--resume" => {
                    args.resume = true;
                    i += 1;
                    continue;
                }
                "--fresh" => {
                    args.fresh = true;
                    i += 1;
                    continue;
                }
                _ => {}
            }
            let value = argv.get(i + 1).map(String::as_str);
            if flag == "--photometric-weight" {
                let v = value.ok_or_else(|| ArgsError::MissingValue(flag.to_owned()))?;
                args.photometric_weight =
                    parse_f64_in(flag, v, (0.0, f64::MAX), "a non-negative number")?;
                photometric_given = true;
                i += 2;
                continue;
            }
            let Some(name) = VALUE_FLAGS.iter().copied().find(|k| *k == flag) else {
                return Err(ArgsError::Unknown(flag.to_owned()));
            };
            let v = value.ok_or_else(|| ArgsError::MissingValue(flag.to_owned()))?;
            args.given.push(name);
            match name {
                "--data" => {
                    args.data = PathBuf::from(v);
                    have_data = true;
                }
                "--out" => args.out = PathBuf::from(v),
                "--size" => {
                    args.size = ModelSize::parse(v).ok_or_else(|| ArgsError::Invalid {
                        flag: name.to_owned(),
                        value: v.to_owned(),
                        expected: "xs|s|m|l|xl|xxl".to_owned(),
                    })?;
                }
                "--strategy" => {
                    args.strategy = TrainStrategy::parse(v).ok_or_else(|| ArgsError::Invalid {
                        flag: name.to_owned(),
                        value: v.to_owned(),
                        expected: "pairwise|batch|align|fusion-align".to_owned(),
                    })?;
                }
                "--epochs" => args.epochs = parse_value(name, v, "a whole number")?,
                "--lr" => {
                    args.lr =
                        parse_f64_in(name, v, (f64::MIN_POSITIVE, f64::MAX), "a positive number")?;
                }
                "--crop" => args.crop = parse_value(name, v, "a whole number")?,
                "--seed" => args.seed = parse_value(name, v, "a whole number")?,
                "--ss-max" => args.ss_max = parse_f64_in(name, v, (0.0, 1.0), "a number in 0..=1")?,
                "--ckpt-every" => {
                    let n: usize = parse_value(name, v, "a whole number")?;
                    args.ckpt_every = n.max(1);
                }
                "--cache-mb" => args.cache_mb = parse_value(name, v, "a whole number")?,
                "--corr-radius" => {
                    let r: usize = parse_value(name, v, "a whole number")?;
                    if r == 0 {
                        return Err(ArgsError::Invalid {
                            flag: name.to_owned(),
                            value: v.to_owned(),
                            expected: "a radius of at least 1".to_owned(),
                        });
                    }
                    args.corr_radius = Some(r);
                }
                "--real-data" => args.real_data = Some(PathBuf::from(v)),
                other => return Err(ArgsError::Unknown(other.to_owned())),
            }
            i += 2;
        }

        if !have_data {
            return Err(ArgsError::Required("--data <dir>"));
        }
        if args.strategy.is_alignment() && args.crop < MIN_ALIGN_CROP {
            return Err(ArgsError::Invalid {
                flag: "--crop".to_owned(),
                value: args.crop.to_string(),
                expected: format!("at least {MIN_ALIGN_CROP} for the alignment strategies"),
            });
        }
        if photometric_given {
            args.given.push("--photometric-weight");
        }
        Ok(args)
    }

    /// The flags that were given but have no effect under [`Self::strategy`]; the
    /// trainer prints a note for each.
    ///
    /// `--ss-max` is pairwise only (the other strategies have no recurrence),
    /// `--corr-radius` belongs to the alignment networks, and `--real-data` /
    /// `--photometric-weight` to `align`.
    #[must_use]
    pub fn ignored_flags(&self) -> Vec<&'static str> {
        let strategy = self.strategy;
        self.given
            .iter()
            .copied()
            .filter(|flag| match *flag {
                "--ss-max" => strategy != TrainStrategy::Pairwise,
                "--corr-radius" => !strategy.is_alignment(),
                "--real-data" | "--photometric-weight" => strategy != TrainStrategy::Align,
                _ => false,
            })
            .collect()
    }
}
