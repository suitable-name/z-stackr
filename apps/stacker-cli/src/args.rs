use std::path::PathBuf;

/// Answers the "what is the stacking target" question.
///
/// For `--input-dir` folders that contain image-bearing subfolders (see
/// [`crate::batch::classify_input_dir`]), passing this flag skips the
/// interactive prompt unconditionally (even on a TTY), and is the only way
/// to resolve that question non-interactively (no TTY + no flag is a hard
/// error).
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub enum StacksMode {
    /// Each direct subfolder of `--input-dir` that contains images is its
    /// own independent stack, run sequentially. `--output-file` is then
    /// interpreted as an output *directory* (see the README's "Batch
    /// processing" section).
    Subfolders,
    /// Only the images directly inside `--input-dir` are the stacking
    /// target (any image-bearing subfolders are ignored) — today's
    /// single-stack, single-output-file behaviour.
    Single,
}

/// Selects which intensity-based optimiser drives subpixel alignment refinement.
///
/// Overrides `optimizer` in the config file (if any). See
/// [`stacker_core::settings::OptimizerSetting`]'s docs for the full
/// Auto/Lucas-Kanade/Nelder-Mead semantics.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub enum OptimizerArg {
    /// Try Lucas-Kanade first, falling back to Nelder-Mead on failure or
    /// RMS regression. The default.
    Auto,
    /// Force the Lucas-Kanade / Gauss-Newton optimiser; no Nelder-Mead
    /// fallback on failure.
    Lk,
    /// Force the original Nelder-Mead bounded simplex optimiser.
    Nm,
}

impl From<OptimizerArg> for stacker_core::settings::OptimizerSetting {
    fn from(arg: OptimizerArg) -> Self {
        match arg {
            OptimizerArg::Auto => Self::Auto,
            OptimizerArg::Lk => Self::LucasKanade,
            OptimizerArg::Nm => Self::NelderMead,
        }
    }
}

#[derive(clap::Parser, Debug, PartialEq, Eq)]
#[command(author, version, about, long_about = None)]
pub struct CliArgs {
    #[arg(long)]
    pub input_dir: PathBuf,

    #[arg(long)]
    pub output_file: PathBuf,

    /// Fusion mode: `apex` (Laplacian pyramid), `relief` (depth map, two
    /// selectable engines via config), `strata` (guided-filter soft-blend
    /// fusion — see `docs/strata-fusion-design.md`), or (with the `nn`
    /// feature) `ai`/`nn`.
    #[arg(long)]
    pub mode: String,

    #[arg(long)]
    pub tile_size: usize,

    /// AI mode (`--mode ai`): name of the model to use (file stem in the models
    /// directory). Defaults to the first discovered model. Only available in
    /// builds with the `nn` feature.
    #[cfg(feature = "nn")]
    #[arg(long)]
    pub model: Option<String>,

    /// AI mode: inference device, `cpu` or `gpu`. Defaults to the best available
    /// (GPU if the binary was built with `--features nn-gpu`, else CPU). Only
    /// available in builds with the `nn` feature.
    #[cfg(feature = "nn")]
    #[arg(long)]
    pub device: Option<String>,

    /// Neural alignment mode (config `alignment_mode = "neural"`): name of the
    /// alignment model to use (file stem in the models directory, filtered to
    /// alignment-capable checkpoints — never interchangeable with the fusion
    /// `--model` above). Defaults to the first discovered alignment model.
    /// Experimental: coarse full-stack neural registration; no pretrained
    /// alignment model ships with this repository yet. Only available in
    /// builds with the `nn` feature.
    #[cfg(feature = "nn")]
    #[arg(long)]
    pub align_model: Option<String>,

    /// Optional path for a structured log file.
    ///
    /// When set, structured tracing events are written to this file in addition
    /// to stdout.  The file is created (or appended) at startup.  If omitted,
    /// only stdout logging is active.
    #[arg(long)]
    pub log_file: Option<PathBuf>,

    /// Monitor the input directory for new files and continuously update the stacked image.
    #[arg(long)]
    pub monitor: bool,

    /// Path to a TOML configuration file.
    /// If the file does not exist, it will be created with default values and comments.
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Resolve the batch-processing question for an `--input-dir` that
    /// contains image-bearing subfolders, without prompting.
    ///
    /// `subfolders` stacks each image-bearing direct subfolder independently
    /// (sequentially) into its own output file inside the `--output-file`
    /// directory; `single` stacks only the images directly inside
    /// `--input-dir`, ignoring any subfolders (today's behaviour).
    ///
    /// If `--input-dir` contains no image-bearing subfolders (today's plain
    /// "one folder of images" case), this flag has no effect. If it does
    /// contain image-bearing subfolders and this flag is omitted: on an
    /// interactive terminal (stdin is a TTY) the CLI prompts for a choice;
    /// otherwise it is a hard error asking you to pass this flag.
    #[arg(long, value_enum)]
    pub stacks: Option<StacksMode>,

    /// Disable the GPU-accelerated compute paths for this run, overriding
    /// `use_gpu` in the config file (if any) — see
    /// `stacker_core::settings::StackingSettings::use_gpu`'s docs. Mirrors
    /// `--monitor`'s plain-switch style: no argument, absent = off. Only
    /// available in builds with the `gpu` feature.
    #[cfg(feature = "gpu")]
    #[arg(long)]
    pub no_gpu: bool,

    /// Select the intensity-based alignment optimiser for this run,
    /// overriding `optimizer` in the config file (if any). `auto` (default)
    /// tries Lucas-Kanade first and falls back to Nelder-Mead on failure or
    /// RMS regression; `lk` forces Lucas-Kanade only (no fallback); `nm`
    /// forces the original Nelder-Mead bounded simplex.
    #[arg(long, value_enum)]
    pub optimizer: Option<OptimizerArg>,
}
