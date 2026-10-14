//! The clap definition of the command line.

use std::path::PathBuf;

use stacker_core::settings::{OptimizerSetting, StackingSettings};

/// Smallest accepted `--tile-size`.
///
/// Every stored tile carries an apron on each side (from 10 px for `relief` up to
/// 512 px for `apex` at the default 8 pyramid levels, see
/// `stacker_pipeline::tiling`), so small tiles multiply the scratch footprint
/// (a 16 px tile with a 128 px apron is stored ~300x larger than its interior)
/// and the file count.
pub const MIN_TILE_SIZE: usize = 64;

fn parse_tile_size(s: &str) -> Result<usize, String> {
    let n: usize = s
        .parse()
        .map_err(|e| format!("'{s}' is not a valid tile size: {e}"))?;
    if n < MIN_TILE_SIZE {
        return Err(format!(
            "tile size must be at least {MIN_TILE_SIZE} pixels (got {n})"
        ));
    }
    Ok(n)
}

/// The inference device named by `--device` (parsed case-insensitively, like
/// `--mode`).
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceRequest {
    /// The `ndarray` CPU backend.
    Cpu,
    /// The wgpu GPU backend (needs a build with `--features nn-gpu`).
    Gpu,
}

/// What `--stacks` decides for an `--input-dir` that holds image-bearing
/// subfolders (see `crate::batch`).
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub enum StacksMode {
    /// Every direct subfolder of `--input-dir` that holds images is a stack of
    /// its own, run one after the other. `--output-file` is then an output
    /// directory.
    Subfolders,
    /// Only the images directly inside `--input-dir` are stacked; subfolders are
    /// ignored (the behaviour without subfolders).
    Single,
}

/// The alignment optimiser named by `--optimizer`; overrides the `optimizer` key
/// of the settings file for the run.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub enum OptimizerArg {
    /// Lucas-Kanade first, the Nelder-Mead simplex when it fails or gets worse.
    Auto,
    /// Lucas-Kanade only.
    Lk,
    /// The Nelder-Mead simplex only.
    Nm,
}

impl From<OptimizerArg> for OptimizerSetting {
    fn from(arg: OptimizerArg) -> Self {
        match arg {
            OptimizerArg::Auto => Self::Auto,
            OptimizerArg::Lk => Self::LucasKanade,
            OptimizerArg::Nm => Self::NelderMead,
        }
    }
}

/// The command line of `z-stackr`: which frames to stack, how to fuse them and
/// where to write the result. `--input-dir`, `--output-file`, `--mode` and
/// `--tile-size` are required; every other option has a default.
#[derive(clap::Parser, Debug, Clone, PartialEq, Eq)]
#[command(author, version, about, long_about = None)]
pub struct CliArgs {
    /// Directory holding the bracketed frames (png, jpg, tif); frames are used
    /// in natural file-name order and the first one is the alignment reference.
    /// With subfolders that hold frames, see `--stacks`.
    #[arg(long)]
    pub input_dir: PathBuf,

    /// Output image. The extension (.png, .tif/.tiff, .jpg/.jpeg) selects the
    /// format; `image_saving.*` in the config sets bit depth and JPEG quality.
    /// With `--mode both` or `--mode all` one file per algorithm is written,
    /// named `<name>_<algorithm>.<ext>`. With `--stacks subfolders` this is the
    /// output directory.
    #[arg(long)]
    pub output_file: PathBuf,

    /// Fusion algorithm, in any letter case: `apex` (Laplacian pyramid), `relief`
    /// (depth map), `strata` (guided fusion), `both` (relief and apex), `all`
    /// (strata, relief and apex) or, in builds with the `nn` feature, `ai`. The
    /// old names `pmax` (apex) and `dmap` (relief) still work.
    #[arg(long)]
    pub mode: String,

    /// Processing tile size in pixels (at least 64; 512 is a good default).
    /// Each frame is stored on disk as padded tiles during the run; for `apex`
    /// the size is rounded up to a multiple of 2^(pyramid_levels-1).
    #[arg(long, value_parser = parse_tile_size)]
    pub tile_size: usize,

    /// AI mode (`--mode ai`): name of the model to use (file stem in the models
    /// directory). Defaults to the first discovered model.
    ///
    /// Only present in builds with the `nn` feature; elsewhere the flag is
    /// rejected instead of being accepted and ignored.
    #[cfg(feature = "nn")]
    #[arg(long)]
    pub model: Option<String>,

    /// AI mode and neural alignment: inference device, `cpu` or `gpu` in any
    /// letter case. Defaults to the best available (GPU if the binary was built
    /// with `--features nn-gpu`, else CPU).
    ///
    /// Only present in builds with the `nn` feature.
    #[cfg(feature = "nn")]
    #[arg(long, value_enum, ignore_case = true)]
    pub device: Option<DeviceRequest>,

    /// Neural alignment (`alignment_mode = "neural"` in the settings file): name
    /// of the alignment model to use (file stem in the models directory, only
    /// alignment models are considered). Defaults to the first discovered
    /// alignment model. Independent of `--model`.
    ///
    /// Only present in builds with the `nn` feature.
    #[cfg(feature = "nn")]
    #[arg(long)]
    pub align_model: Option<String>,

    /// Keep this run off the graphics card: the heavy stages run on the CPU
    /// whatever `compute_device` says, and the neural models run on the CPU too.
    /// The same as `compute_device = "cpu"` and `--device cpu`.
    ///
    /// Only present in builds with the `gpu` or `nn-gpu` feature.
    #[cfg(any(feature = "gpu", feature = "nn-gpu"))]
    #[arg(long)]
    pub no_gpu: bool,

    /// Alignment optimiser for this run, overriding `optimizer` in the settings
    /// file: `auto` (Lucas-Kanade, the simplex as fallback), `lk` (Lucas-Kanade
    /// only) or `nm` (Nelder-Mead simplex only). Any letter case.
    #[arg(long, value_enum, ignore_case = true)]
    pub optimizer: Option<OptimizerArg>,

    /// What to do when `--input-dir` holds subfolders with images: `subfolders`
    /// stacks every such subfolder on its own into `--output-file` (a directory;
    /// the files are named by `image_saving.filename_template`), `single` stacks
    /// only the images directly inside `--input-dir`. Without the flag a terminal
    /// is asked; without a terminal the run stops and asks for the flag. No
    /// effect when no subfolder holds images. Not combinable with `--monitor` in
    /// `subfolders` mode.
    #[arg(long, value_enum, ignore_case = true)]
    pub stacks: Option<StacksMode>,

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

    /// Path to a TOML configuration file (see `--write-default-config`).
    /// The file must exist and parse; a parse error stops the run.
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Create the `--config` file with commented default values when it does
    /// not exist yet (otherwise a missing file is an error).
    #[arg(long, requires = "config")]
    pub write_default_config: bool,
}

/// The fusion-algorithm labels that are appended to the output name when one run
/// writes several files (`--mode both` or `all`).
const PASS_LABELS: [&str; 4] = ["apex", "relief", "strata", "ai"];

impl CliArgs {
    /// Paths the pipeline itself writes (output image, log file).  They are
    /// never treated as input frames, even when they sit in the input directory.
    ///
    /// With `--mode both` or `all` the files are named `<name>_<algorithm>.<ext>`;
    /// those names are listed too, so a result next to the frames is not stacked
    /// again and does not retrigger the `--monitor` loop.
    #[must_use]
    pub fn own_output_paths(&self) -> Vec<PathBuf> {
        let mut paths = vec![self.output_file.clone()];
        paths.extend(
            PASS_LABELS
                .iter()
                .filter_map(|label| crate::pipeline::suffixed_path(&self.output_file, label)),
        );
        if let Some(log) = &self.log_file {
            paths.push(log.clone());
        }
        paths
    }

    /// The settings of the run: `settings` with the command-line overrides applied
    /// (`--optimizer`, `--no-gpu`). The file is never written back.
    #[must_use]
    pub fn effective_settings(&self, settings: &StackingSettings) -> StackingSettings {
        let mut effective = settings.clone();
        if let Some(optimizer) = self.optimizer {
            effective.optimizer = optimizer.into();
        }
        #[cfg(any(feature = "gpu", feature = "nn-gpu"))]
        if self.no_gpu {
            effective.compute_device = stacker_core::settings::ComputeDevice::Cpu;
        }
        effective
    }

    /// The inference device the neural models run on: `--device`, else the CPU
    /// for `--no-gpu`, else `None` (the best the build offers).
    ///
    /// # Errors
    /// `--no-gpu` together with `--device gpu`.
    #[cfg(feature = "nn")]
    pub fn inference_device(
        &self,
    ) -> Result<Option<DeviceRequest>, stacker_core::error::StackerError> {
        #[cfg(any(feature = "gpu", feature = "nn-gpu"))]
        if self.no_gpu {
            return match self.device {
                None | Some(DeviceRequest::Cpu) => Ok(Some(DeviceRequest::Cpu)),
                Some(DeviceRequest::Gpu) => Err(stacker_core::error::StackerError::Config(
                    "--no-gpu and --device gpu contradict each other".to_owned(),
                )),
            };
        }
        Ok(self.device)
    }

    /// A command line for the tests: required options only, everything else off.
    #[cfg(test)]
    pub(crate) fn for_test(
        input_dir: &std::path::Path,
        output_file: &std::path::Path,
        mode: &str,
        tile_size: usize,
    ) -> Self {
        Self {
            input_dir: input_dir.to_path_buf(),
            output_file: output_file.to_path_buf(),
            mode: mode.to_owned(),
            tile_size,
            #[cfg(feature = "nn")]
            model: None,
            #[cfg(feature = "nn")]
            device: None,
            #[cfg(feature = "nn")]
            align_model: None,
            #[cfg(any(feature = "gpu", feature = "nn-gpu"))]
            no_gpu: false,
            optimizer: None,
            stacks: None,
            log_file: None,
            monitor: false,
            config: None,
            write_default_config: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn optimizer_argument_maps_onto_the_setting() {
        assert_eq!(
            OptimizerSetting::from(OptimizerArg::Auto),
            OptimizerSetting::Auto
        );
        assert_eq!(
            OptimizerSetting::from(OptimizerArg::Lk),
            OptimizerSetting::LucasKanade
        );
        assert_eq!(
            OptimizerSetting::from(OptimizerArg::Nm),
            OptimizerSetting::NelderMead
        );
    }

    #[test]
    fn effective_settings_applies_the_optimizer_override_only() {
        let mut args = CliArgs::for_test(Path::new("in"), Path::new("out.png"), "apex", 128);
        let base = StackingSettings {
            optimizer: OptimizerSetting::Auto,
            ..StackingSettings::default()
        };
        assert_eq!(args.effective_settings(&base), base, "no flag, no change");

        args.optimizer = Some(OptimizerArg::Nm);
        let effective = args.effective_settings(&base);
        assert_eq!(effective.optimizer, OptimizerSetting::NelderMead);
        let restored = StackingSettings {
            optimizer: OptimizerSetting::Auto,
            ..effective
        };
        assert_eq!(restored, base, "only the optimizer changed");
    }

    #[cfg(any(feature = "gpu", feature = "nn-gpu"))]
    #[test]
    fn no_gpu_forces_the_cpu() {
        use stacker_core::settings::ComputeDevice;

        let mut args = CliArgs::for_test(Path::new("in"), Path::new("out.png"), "apex", 128);
        let gpu = StackingSettings {
            compute_device: ComputeDevice::Gpu,
            ..StackingSettings::default()
        };
        assert_eq!(
            args.effective_settings(&gpu).compute_device,
            ComputeDevice::Gpu
        );
        args.no_gpu = true;
        assert_eq!(
            args.effective_settings(&gpu).compute_device,
            ComputeDevice::Cpu
        );
    }

    #[test]
    fn own_output_paths_include_the_per_algorithm_names() {
        let args = CliArgs::for_test(Path::new("in"), Path::new("in/result.png"), "all", 128);
        let own = args.own_output_paths();
        for name in [
            "result.png",
            "result_apex.png",
            "result_relief.png",
            "result_strata.png",
        ] {
            assert!(own.iter().any(|p| p.ends_with(name)), "{name} in {own:?}");
        }
    }
}
