//! The CLI's out-of-core run: validate, align into a tile store, auto-cull,
//! fuse tile by tile, crop and save.
//!
//! Every per-frame and per-tile operation is a shared stage of
//! `stacker_pipeline` (decode, alignment, cull, fuse, encode), and the tile loop
//! itself is `stacker_pipeline::tiled`; what is left here is the request
//! (algorithms, output names), the progress bars and the output path rules.

mod fuse;
#[cfg(feature = "nn")]
mod neural;
pub(crate) mod output;

#[cfg(test)]
mod behaviour_tests;
#[cfg(test)]
mod tests;

pub use fuse::{Engine, FusionMode, renamed_mode};
pub use stacker_pipeline::order::{has_frame_extension, is_excluded};

use indicatif::{ProgressBar, ProgressStyle};
use stacker_core::{error::StackerError, memory::MmapTileManager, settings::StackingSettings};
use stacker_pipeline::{
    NeverCancel, Progress, Stage,
    compute::Compute,
    order::{collect_image_paths, select_frames},
    output::{EncodedImage, source_exif, supports_exif},
    session::{FrameSource, PathSource},
    tiled::{
        FusePlan, Hooks, ScratchDir, StoreFrames, TiledPlan, TracingLog,
        align_into_store_with_model, crop_rect, frame_order, fuse_tiles, plan_run,
    },
    tiling::{Method, apron_for, plan_tiles},
};
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    time::Instant,
};

use crate::args::{CliArgs, MIN_TILE_SIZE};
#[cfg(feature = "nn")]
use fuse::{NnModel, load_nn_model, nn_fuse_tile};
use output::{OutputPlan, check_output_writable, resolve_output};

/// An error for a request that cannot be honoured (bad option, bad input set).
const fn invalid_input(msg: String) -> StackerError {
    StackerError::Validation(msg)
}

/// `path` with `_<label>` appended to the file stem, keeping the directory and the
/// extension (`out/result.png` and `apex` give `out/result_apex.png`); `None` for a
/// path without a file name.
///
/// This is how a run that writes several algorithms (`--mode both` or `all`)
/// names its files.
#[must_use]
pub fn suffixed_path(path: &Path, label: &str) -> Option<PathBuf> {
    let stem = path.file_stem()?.to_string_lossy();
    let name = path.extension().map_or_else(
        || format!("{stem}_{label}"),
        |ext| format!("{stem}_{label}.{}", ext.to_string_lossy()),
    );
    Some(path.with_file_name(name))
}

/// Steps of a progress bar: the stages report fractions, not counts.
const BAR_STEPS: u16 = 1000;

fn bar_style(colors: &str) -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(&format!(
            "{{spinner:.green}} [{{elapsed_precise}}] [{{bar:40.{colors}}}] {{percent:>3}}% \
             ({{eta}}) {{msg}}"
        ))
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("#>-")
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a bar position: a rounded fraction scaled into 0..=BAR_STEPS"
)]
fn bar_position(fraction: f32) -> u64 {
    (fraction.clamp(0.0, 1.0) * f32::from(BAR_STEPS)).round() as u64
}

/// Shows the alignment and the fusion stage on indicatif bars. A bar is created
/// when its stage first reports and finished when it reaches 1; a run that fuses
/// several algorithms therefore gets one fusion bar each.
struct Bars {
    align: RefCell<Option<ProgressBar>>,
    fuse: RefCell<Option<ProgressBar>>,
    fuse_message: RefCell<String>,
}

impl Bars {
    fn new() -> Self {
        Self {
            align: RefCell::new(None),
            fuse: RefCell::new(None),
            fuse_message: RefCell::new("Fusing tiles".to_owned()),
        }
    }

    /// The next fusion bar names `engine`.
    fn begin_fuse(&self, engine: Engine) {
        *self.fuse_message.borrow_mut() = format!("Fusing tiles ({})", engine.label());
    }
}

impl Progress for Bars {
    fn stage(&self, stage: Stage, fraction: f32, _label: &str) {
        let (slot, message, colors, done) = match stage {
            Stage::Align => (
                &self.align,
                "Aligning frames".to_owned(),
                "cyan/blue",
                "Alignment complete",
            ),
            Stage::Fuse => (
                &self.fuse,
                self.fuse_message.borrow().clone(),
                "magenta/blue",
                "Fusion complete",
            ),
            Stage::Load | Stage::Cull | Stage::Save => return,
        };
        let mut slot = slot.borrow_mut();
        let bar = slot.get_or_insert_with(|| {
            let bar = ProgressBar::new(u64::from(BAR_STEPS));
            bar.set_style(bar_style(colors));
            bar.set_message(message);
            bar
        });
        bar.set_position(bar_position(fraction));
        if fraction >= 1.0
            && let Some(bar) = slot.take()
        {
            bar.finish_with_message(done);
        }
    }
}

/// What is known and checked before any frame is decoded.
struct Prepared {
    mode: FusionMode,
    paths: Vec<PathBuf>,
    /// One output per engine of the mode, in run order.
    outputs: Vec<(Engine, OutputPlan)>,
}

/// Validate the request: mode, tile size, frame set, output paths.  Nothing in
/// here decodes an image or touches the scratch directory, so a mistake costs
/// milliseconds instead of a finished alignment pass.
fn prepare_run(args: &CliArgs, settings: &StackingSettings) -> Result<Prepared, StackerError> {
    let mode: FusionMode = args.mode.parse()?;
    if let Some(current) = renamed_mode(&args.mode) {
        tracing::warn!(
            "--mode {} is the old name of --mode {current}; use the new name",
            args.mode
        );
    }
    if args.tile_size < MIN_TILE_SIZE {
        return Err(invalid_input(format!(
            "tile size must be at least {MIN_TILE_SIZE} pixels (got {})",
            args.tile_size
        )));
    }
    #[cfg(not(feature = "nn"))]
    if settings.alignment_mode == stacker_core::settings::AlignmentModeSetting::Neural {
        return Err(StackerError::Config(
            "alignment_mode = \"neural\" needs a build with --features nn".to_owned(),
        ));
    }

    tracing::info!("collecting images from {}", args.input_dir.display());
    let paths = collect_image_paths(&args.input_dir, &args.own_output_paths())?;
    let paths = select_frames(paths, settings);
    let first_stem = paths[0].file_stem().map_or_else(
        || "stacked".to_owned(),
        |s| s.to_string_lossy().into_owned(),
    );

    let base = resolve_output(&args.output_file, &settings.image_saving, &first_stem)?;
    if settings.image_saving.copy_metadata && !supports_exif(base.format) {
        tracing::warn!(
            "image_saving.copy_metadata: a TIFF file cannot carry the EXIF block of the source; \
             the result has none (use PNG or JPEG to keep it)"
        );
    }
    let engines = mode.engines();
    let outputs: Vec<(Engine, OutputPlan)> = engines
        .iter()
        .map(|&engine| {
            let mut plan = base.clone();
            if engines.len() > 1
                && let Some(path) = suffixed_path(&base.path, engine.label())
            {
                plan.path = path;
            }
            (engine, plan)
        })
        .collect();

    // The resolved names can differ from `--output-file` (template, default
    // directory, per-algorithm suffix); keep a previous result under any of them
    // out of the stack too.
    let own: Vec<PathBuf> = std::iter::once(base.path.clone())
        .chain(outputs.iter().map(|(_, plan)| plan.path.clone()))
        .collect();
    let paths: Vec<PathBuf> = paths
        .into_iter()
        .filter(|p| !is_excluded(p, &own))
        .collect();
    if paths.is_empty() {
        return Err(invalid_input(format!(
            "no input frames left in '{}' after excluding the output file",
            args.input_dir.display()
        )));
    }
    for (_, plan) in &outputs {
        check_output_writable(plan)?;
    }
    tracing::info!(
        n_frames = paths.len(),
        outputs = outputs.len(),
        output = %base.path.display(),
        "found images"
    );
    Ok(Prepared {
        mode,
        paths,
        outputs,
    })
}

/// The tile grid of a run over `size` pixels fusing with `engines`.
///
/// One engine gets its own geometry. Several share one tile store, so the grid
/// is the `Apex` one (the tile size rounded for the pyramid) with the largest
/// apron any of them needs.
fn plan_for(
    engines: &[Engine],
    size: (usize, usize),
    requested_tile: usize,
    settings: &StackingSettings,
) -> (TiledPlan, Method) {
    if let [only] = engines {
        let method = only.method();
        return (plan_run(method, size, requested_tile, settings), method);
    }
    let mut plan = plan_run(Method::Apex, size, requested_tile, settings);
    let apron = engines
        .iter()
        .map(|engine| apron_for(engine.method(), settings))
        .max()
        .unwrap_or(plan.geometry.apron);
    plan.geometry.apron = apron;
    plan.tiles = plan_tiles(size.0, size.1, plan.geometry.tile_size, apron);
    (plan, Method::Apex)
}

/// The tiles on disk. The store is closed before its directory is removed (a
/// mapped file cannot be deleted on Windows), which the field order guarantees.
struct TileStore {
    store: MmapTileManager,
    _scratch: ScratchDir,
}

/// Run the full focus-stacking pipeline in a **tiled, out-of-core** manner.
///
/// ## Memory model
///
/// The per-pixel figures of every stage are in the `stacker-pipeline` README
/// (section "Memory and disk: per-pixel figures"). In short:
///
/// - **Alignment pre-pass**: one frame is decoded, aligned, committed to the
///   tile store and dropped at a time; RAM is about 39 bytes per pixel of one
///   frame (frame, reference, working buffers), independent of the number of
///   frames. The alignment itself needs the whole frame: matching and
///   registration on tiles would give incoherent, tile-local transforms.
/// - **Tile store**: `12 · ((T + 2A) / T)²` bytes of disk per source pixel per
///   frame, where `T` is the tile size and `A` the apron.
/// - **Fusion**: one padded tile of every frame in RAM at a time
///   (`N · (12 + pyramid or focus map) · (T + 2A)²` bytes), plus the output
///   buffer at its bit depth (3 or 6 bytes per pixel) and, for `Relief`, the
///   whole-image focus maximum (4 bytes per pixel; the multigrid solver adds the
///   planes described in `stacker_pipeline::tiled`).
///
/// ## Apron strategy
///
/// Each stored tile carries an apron that covers what the fusion method reads
/// (see `stacker_pipeline::tiling`): `Apex` the dependence radius of its
/// Laplacian pyramid, derived from `pyramid_levels`; `Relief` the focus and
/// guided-filter radii; `Strata` its two filter radii; AI the network's
/// receptive field. Only the interior is copied into the output after fusion,
/// and the interior equals what a whole-image run would produce. A run that
/// fuses several algorithms (`both`, `all`) aligns once, stores tiles with the
/// largest apron and fuses the same store once per algorithm.
///
/// # Errors
/// Returns [`StackerError`] on I/O failures, unsupported mode strings, an
/// unusable output path, a frame whose size differs from the reference, or
/// alignment failures propagated from image I/O.
pub fn run_pipeline(args: &CliArgs, settings: &StackingSettings) -> Result<(), StackerError> {
    run_pipeline_in(args, settings, &std::env::temp_dir())
}

/// [`run_pipeline`] with an explicit scratch root for the temporary tile
/// directory (the system temp directory in production).  The directory is
/// removed on every exit path, success or error.
///
/// # Errors
/// Same as [`run_pipeline`].
pub fn run_pipeline_in(
    args: &CliArgs,
    settings: &StackingSettings,
    scratch_root: &Path,
) -> Result<(), StackerError> {
    run_stack_in(args, settings, scratch_root).map(drop)
}

/// [`run_pipeline_in`] that returns the files it wrote, one per algorithm of
/// `--mode`, in run order.
///
/// # Errors
/// Same as [`run_pipeline`].
pub fn run_stack_in(
    args: &CliArgs,
    settings: &StackingSettings,
    scratch_root: &Path,
) -> Result<Vec<PathBuf>, StackerError> {
    // `--optimizer` and `--no-gpu` apply to this run only.
    let settings = &args.effective_settings(settings);
    let prepared = prepare_run(args, settings)?;
    run_prepared(args, settings, prepared, scratch_root)
}

/// The run itself, after validation: reference frame, tile grid, alignment into
/// the store, fusion order, then one fusion, crop and save per engine.
#[expect(
    clippy::too_many_lines,
    reason = "one linear run sequence (reference, grid, alignment, order, fuse and save per \
              engine) whose steps share a dozen locals"
)]
fn run_prepared(
    args: &CliArgs,
    settings: &StackingSettings,
    prepared: Prepared,
    scratch_root: &Path,
) -> Result<Vec<PathBuf>, StackerError> {
    let t_total = Instant::now();
    let Prepared {
        mode,
        paths,
        outputs,
    } = prepared;
    let n_frames = paths.len();
    tracing::info!(?mode, n_frames, "stacking");

    // The AI model and the alignment model are loaded (or the run fails) before
    // any alignment work.
    #[cfg(feature = "nn")]
    let nn_model: Option<NnModel> = if matches!(mode, FusionMode::Ai) {
        Some(load_nn_model(args)?)
    } else {
        None
    };
    #[cfg(feature = "nn")]
    let align_model = neural::align_model_for(args, settings)?;
    #[cfg(not(feature = "nn"))]
    let align_model: Option<stacker_pipeline::align::AlignModel> = None;

    // ── 1. Reference frame: image dimensions and tile grid ───────────────────
    let first_frame = paths[0].clone();
    let mut source = PathSource::new(paths);
    let t_load = Instant::now();
    let reference = source.decode(0, &settings.preprocessing)?;
    let size = (reference.width, reference.height);
    if size.0 == 0 || size.1 == 0 {
        return Err(invalid_input(format!(
            "reference frame '{}' has unusable dimensions {}x{}",
            first_frame.display(),
            size.0,
            size.1
        )));
    }
    tracing::info!(
        img_w = size.0,
        img_h = size.1,
        elapsed_ms = t_load.elapsed().as_millis(),
        "reference frame loaded"
    );

    let engines: Vec<Engine> = outputs.iter().map(|(engine, _)| *engine).collect();
    let (plan, plan_method) = plan_for(&engines, size, args.tile_size, settings);
    tracing::info!(
        "image {}x{}, tile_size={}, apron={} px, {} tiles, scratch about {} bytes for {n_frames} \
         frames",
        size.0,
        size.1,
        plan.geometry.tile_size,
        plan.geometry.apron,
        plan.tiles.len(),
        plan.scratch_bytes(n_frames),
    );
    if let Some(warning) = plan.padding_warning(plan_method) {
        tracing::warn!("{warning}");
    }

    // ── 2. Alignment + tile-commit pre-pass ──────────────────────────────────
    //
    // The guard owns the scratch directory: whatever happens below (an error, a
    // panic, success) the tiles are removed when it drops.
    let scratch = ScratchDir::create(scratch_root)?;
    let mut store = MmapTileManager::new(scratch.path().to_path_buf());

    let bars = Bars::new();
    let hooks = Hooks {
        progress: &bars,
        cancel: &NeverCancel,
        log: &TracingLog,
    };
    // The objective, the warp and the fusion run on the device the settings ask
    // for (the CPU unless `compute_device = "gpu"` in a build with GPU support).
    let mut compute = Compute::for_settings(settings);

    tracing::info!("pre-pass: aligning and tiling {n_frames} frames");
    let t_align = Instant::now();
    let coverage = align_into_store_with_model(
        &mut source,
        &reference,
        settings,
        align_model,
        &plan.tiles,
        &mut store,
        &mut compute,
        &hooks,
    )?;
    drop(reference);
    tracing::info!(
        elapsed_ms = t_align.elapsed().as_millis(),
        "alignment + tile-commit pre-pass complete"
    );

    // ── 3. Auto-cull and order (settings.auto_cull, sort_by_sharpness) ───────
    hooks.progress.stage(Stage::Cull, 0.0, "");
    let order = frame_order(settings, &plan, &store, n_frames, &hooks)?;
    hooks.progress.stage(Stage::Cull, 1.0, "");

    // The crop is the same for every engine: it comes from the alignment.
    let crop = crop_rect(&coverage, settings, size, |i| source.name(i).to_owned());
    let exif_source = first_frame;

    // ── 4. One fusion, crop and save per engine ──────────────────────────────
    tracing::info!(
        "fusion pass: {} tiles × {} frames, {} algorithm(s)",
        plan.tiles.len(),
        order.len(),
        outputs.len()
    );
    let mut disk = Some(TileStore {
        store,
        _scratch: scratch,
    });
    let mut written = Vec::with_capacity(outputs.len());
    let last = outputs.len().saturating_sub(1);
    for (position, (engine, out_plan)) in outputs.into_iter().enumerate() {
        let Some(tiles) = disk.as_ref() else {
            return Err(invalid_input("the tile store is gone".to_owned()));
        };
        let t_fuse = Instant::now();
        let mut out = EncodedImage::zeroed(
            size.0,
            size.1,
            out_plan.format,
            out_plan.bit16,
            out_plan.jpeg_quality,
        );
        bars.begin_fuse(engine);
        let fuse_plan = FusePlan::new(engine.method(), settings, &plan, &order);
        let frames = StoreFrames(&tiles.store);
        match engine {
            #[cfg(feature = "nn")]
            Engine::Ai => {
                let nn = nn_model
                    .as_ref()
                    .ok_or_else(|| invalid_input("the AI model is not loaded".to_owned()))?;
                stacker_pipeline::tiled::fuse_tiles_by(
                    &fuse_plan,
                    &frames,
                    &mut out,
                    &hooks,
                    |_, tile_frames| {
                        nn_fuse_tile(nn, &tile_frames)
                            .map_err(stacker_pipeline::PipelineError::Core)
                    },
                )?;
            }
            Engine::Strata | Engine::Relief | Engine::Apex => {
                fuse_tiles(&fuse_plan, &frames, &mut compute, &mut out, &hooks)?;
            }
        }
        if let Some(notice) = compute.take_notice() {
            tracing::warn!("{notice}");
        }
        tracing::info!(
            engine = engine.label(),
            elapsed_ms = t_fuse.elapsed().as_millis(),
            "fusion pass complete"
        );
        if position == last {
            // The tiles are no longer needed; release the disk before encoding.
            disk = None;
        }

        // ── 5. Crop to the common valid area and save the output ─────────────
        let t_encode = Instant::now();
        let exif = if supports_exif(out_plan.format) {
            source_exif(
                &exif_source,
                &settings.preprocessing,
                &settings.image_saving,
            )
        } else {
            None
        };
        let mut out = out.with_exif(exif);
        if let Some(rect) = crop {
            tracing::info!(
                cx = rect.0,
                cy = rect.1,
                cw = rect.2,
                ch = rect.3,
                original_w = size.0,
                original_h = size.1,
                "cropping fused output to common valid area"
            );
            out = out.crop(rect);
            // The same rule as `TiledOutput::finish`: after a crop that removed
            // something, `resize_cropped_to_original` resamples back to the canvas.
            if settings.resize_cropped_to_original {
                out = out.resize(size.0, size.1);
            }
        }
        out.save(&out_plan.path)?;
        tracing::info!(
            output_path = %out_plan.path.display(),
            format = ?out_plan.format,
            encode_ms = t_encode.elapsed().as_millis(),
            "output written"
        );
        written.push(out_plan.path);
    }
    tracing::info!(
        total_ms = t_total.elapsed().as_millis(),
        "pipeline complete"
    );

    Ok(written)
}
