//! The stacking flows behind the Python functions, in plain Rust.
//!
//! Everything here is Python-free: the numpy layer (`crate::array_api`, `crate::file_api`)
//! only converts its arguments and results around these functions, which is also why they are
//! tested without an interpreter.
//!
//! | Flow | Function | Built on |
//! |---|---|---|
//! | frames in RAM (`stack_arrays`) | [`stack_in_ram`] | `session` |
//! | files, out of core (`tile_size > 0`) | [`run_files`] | `tiled::run_tiled` |
//! | files, in RAM (`tile_size = 0`) | [`run_files`] | `session::load_frames`, [`stack_in_ram`] |
//! | one stack per sub-folder (`batch_stack`) | [`run_batch`] | [`run_files`] |
//!
//! (`session` and `tiled` are modules of `stacker_pipeline`.)
//!
//! All of them report through the `Hooks` of `stacker_pipeline::tiled` (progress, cancel,
//! log), so one adapter in `crate::bridge` serves every function.
//!
//! # Behaviour that follows the settings, in both file flows and the array flow
//!
//! `auto_cull`, `auto_cull_threshold_pct` and `sort_by_sharpness` decide which frames are fused
//! and in what order (in the out-of-core flow from the focus census of the tile store).
//! `crop_to_common_area` crops to the area every aligned frame covers;
//! `resize_cropped_to_original` stretches the result back, except in the out-of-core flow,
//! which keeps the cropped size (the result would have to be resampled in full).

use std::{
    cell::RefCell,
    path::{Path, PathBuf},
};

use stacker_algo::strata::NO_PREVIEW;
use stacker_core::{
    image::PlanarImage,
    settings::{ImageSavingSettings, StackingSettings},
};
use stacker_pipeline::{
    PipelineError, Stage,
    compute::Compute,
    crop::resize_planar,
    fraction_of,
    load::{EXIF_HEAD_BYTES, apply_orientation, has_raw_extension, jpeg_exif_orientation},
    order::{collect_image_paths, is_excluded, natural_sort, select_frames},
    output::{EncodedImage, encode, jpeg_quality, source_exif, supports_exif, wants_16_bit},
    session::{self, PathSource, load_frames},
    tiled::{Hooks, RunLog, TiledJob, TiledOutput, effective_tile_size, run_tiled},
    tiling::Method,
};

use crate::options::{ArgError, extension_of, output_format_for_path};

// ── the fusion method ───────────────────────────────────────────────────────

/// The fusion methods Python can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuseMode {
    /// Laplacian-pyramid fusion.
    Apex,
    /// Depth-map fusion (guided filter, or multigrid with `relief_use_multigrid`).
    Relief,
    /// Guided-filter soft blend.
    Strata,
}

impl FuseMode {
    /// The labels [`Self::parse`] accepts (the older `"pmax"` and `"dmap"` work as well).
    pub const OPTIONS: [&'static str; 3] = ["apex", "relief", "strata"];

    /// Parse a `mode` string (any letter case).
    ///
    /// # Errors
    /// [`ArgError`] listing the valid options for anything else; `"ai"` has its own message
    /// because the neural fusion is not available from Python.
    pub fn parse(s: &str) -> Result<Self, ArgError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "apex" | "pmax" => Ok(Self::Apex),
            "relief" | "dmap" => Ok(Self::Relief),
            "strata" => Ok(Self::Strata),
            "ai" => Err(ArgError::new(format!(
                "mode \"ai\" is not available from Python; valid options are: {}",
                Self::OPTIONS.join(", ")
            ))),
            _ => Err(ArgError::new(format!(
                "invalid mode {s:?}; valid options are: {}",
                Self::OPTIONS.join(", ")
            ))),
        }
    }

    /// The label of this method.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Apex => Self::OPTIONS[0],
            Self::Relief => Self::OPTIONS[1],
            Self::Strata => Self::OPTIONS[2],
        }
    }

    /// The method of the tile geometry.
    #[must_use]
    pub const fn method(self) -> Method {
        match self {
            Self::Apex => Method::Apex,
            Self::Relief => Method::Relief,
            Self::Strata => Method::Strata,
        }
    }
}

// ── the log ─────────────────────────────────────────────────────────────────

/// A [`RunLog`] that keeps the warnings, so the Python layer can raise them as
/// `warnings.warn` once it holds the GIL again. Information messages are dropped.
#[derive(Debug, Default)]
pub struct CollectLog {
    warnings: RefCell<Vec<String>>,
}

impl CollectLog {
    /// The warnings collected so far, in order.
    #[must_use]
    pub fn into_warnings(self) -> Vec<String> {
        self.warnings.into_inner()
    }
}

impl RunLog for CollectLog {
    fn info(&self, _message: &str) {}

    fn warn(&self, message: &str) {
        self.warnings.borrow_mut().push(message.to_owned());
    }
}

// ── selecting frames ────────────────────────────────────────────────────────

/// The positions of the frames a run uses out of `count`.
///
/// Reversed with `sort_reverse`, then every `every_nth`-th (`1` keeps all). The rule of
/// `stacker_pipeline::order::select_frames` for frames that are not files. The first frame of
/// the (reversed) order is always kept.
#[must_use]
pub fn select_indices(count: usize, sort_reverse: bool, every_nth: u32) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..count).collect();
    if sort_reverse {
        indices.reverse();
    }
    let nth = every_nth.max(1) as usize;
    indices
        .into_iter()
        .enumerate()
        .filter(|(position, _)| position % nth == 0)
        .map(|(_, index)| index)
        .collect()
}

// ── frames in RAM ───────────────────────────────────────────────────────────

/// The most frames `Strata` fuses.
const MAX_STRATA_FRAMES: usize = 65_536;

/// Stack frames that are already in memory: align to `frames[0]`, cull and order, fuse, crop.
///
/// The steps and the settings they read:
///
/// 1. alignment: `alignment_mode`, `optimizer`, `bounded_search`, `akaze_seeding`,
///    `correct_brightness`, `chain_alignment`, `compute_device`;
/// 2. the crop to the area every frame covers (`crop_to_common_area`; refused below half the
///    frame);
/// 3. cull and order: `auto_cull`, `auto_cull_threshold_pct`, `sort_by_sharpness`;
/// 4. fusion: `pyramid_levels`, `use_all_color_channels`, `grit_suppression` (`Apex`),
///    `relief_*` (`Relief`), `strata_*` (`Strata`);
/// 5. `resize_cropped_to_original`.
///
/// The preprocessing settings are not applied (the caller owns the pixels) and neither are
/// `stack_every_nth` and `sort_reverse`; [`select_indices`] does the latter two.
///
/// # Errors
/// [`PipelineError::Cancelled`] when the hooks' cancel fires; [`PipelineError::Invalid`] for
/// no frame, frames of different sizes, a `Strata` stack of more than 65 536 frames or a cull
/// that leaves no frame.
pub fn stack_in_ram(
    mut frames: Vec<PlanarImage<f32>>,
    settings: &StackingSettings,
    mode: FuseMode,
    hooks: &Hooks<'_>,
) -> Result<PlanarImage<f32>, PipelineError> {
    let Some(first) = frames.first() else {
        return Err(PipelineError::invalid(
            "No images to process: add some images first.",
        ));
    };
    let (width, height) = (first.width, first.height);
    for (index, frame) in frames.iter().enumerate().skip(1) {
        if (frame.width, frame.height) != (width, height) {
            return Err(PipelineError::invalid(format!(
                "frame {index} is {}x{} px but frame 0 is {width}x{height} px; all frames must \
                 have the same size.",
                frame.width, frame.height
            )));
        }
    }
    if mode == FuseMode::Strata && frames.len() > MAX_STRATA_FRAMES {
        return Err(PipelineError::invalid(format!(
            "Strata fuses at most {MAX_STRATA_FRAMES} frames, but the stack has {}.",
            frames.len()
        )));
    }
    let count = frames.len();
    let mut compute = Compute::for_settings(settings);

    // Align every frame to the first.
    hooks.progress.stage(Stage::Align, 0.0, "");
    let aligned_total = count.saturating_sub(1);
    let progress = hooks.progress;
    let mut on_frame = |index: usize, _: &PlanarImage<f32>, _: &[bool]| {
        progress.stage(Stage::Align, fraction_of(index, aligned_total), "");
    };
    let outcome = session::align_frames(
        &mut frames,
        settings,
        &mut compute,
        hooks.cancel,
        &mut on_frame,
    )
    .ok_or(PipelineError::Cancelled)?;
    let names: Vec<String> = (0..count).map(|i| format!("frame {i}")).collect();
    if let Some(warning) = outcome.warning(&names) {
        hooks.log.warn(&warning);
    }
    let crop = if settings.crop_to_common_area {
        outcome.crop
    } else {
        None
    };
    hooks.progress.stage(Stage::Align, 1.0, "");
    check_cancel(hooks)?;

    // Cull and order.
    if count > 1 && (settings.auto_cull || settings.sort_by_sharpness) {
        hooks.progress.stage(Stage::Cull, 0.0, "");
        let decision = session::cull(&frames, settings)?;
        if let Some(warning) = decision.warning() {
            hooks.log.warn(&warning);
        }
        frames = session::reorder(frames, &decision.order);
        hooks.progress.stage(Stage::Cull, 1.0, "");
        check_cancel(hooks)?;
    }

    // Fuse.
    hooks.progress.stage(Stage::Fuse, 0.0, mode.name());
    let fused = match mode {
        FuseMode::Apex => session::fuse_apex(&mut compute, &frames, settings),
        FuseMode::Relief => session::fuse_relief(&mut compute, &frames, settings, |focus_max| {
            Some(session::threshold_from_settings(focus_max, settings))
        })
        .ok_or(PipelineError::Cancelled)?,
        FuseMode::Strata => {
            session::fuse_strata(&frames, settings, hooks.progress, NO_PREVIEW, |_| {})
        }
    };
    drop(frames);
    if let Some(notice) = compute.take_notice() {
        hooks.log.warn(&notice);
    }
    hooks.progress.stage(Stage::Fuse, 1.0, "");

    // Crop, and stretch back if asked.
    let fused = session::crop_result(fused, crop);
    if crop.is_some() && settings.resize_cropped_to_original {
        return Ok(resize_planar(fused, width, height));
    }
    Ok(fused)
}

/// `Err(Cancelled)` once the hooks' cancel has fired.
fn check_cancel(hooks: &Hooks<'_>) -> Result<(), PipelineError> {
    if hooks.cancel.is_cancelled() {
        Err(PipelineError::Cancelled)
    } else {
        Ok(())
    }
}

// ── files ───────────────────────────────────────────────────────────────────

/// What one [`run_files`] run is asked to do.
#[derive(Debug, Clone, Copy)]
pub struct FileJob<'a> {
    /// The input frames, in stack order.
    pub paths: &'a [PathBuf],
    /// The result file; its extension picks the format.
    pub output_file: &'a Path,
    /// The fusion method.
    pub mode: FuseMode,
    /// Edge of the out-of-core tiles; `0` takes `settings.tile_size`, and `0` there as well
    /// runs the stack in RAM.
    pub tile_size: usize,
}

/// Stack the frames of `job` into `job.output_file`.
///
/// `settings.stack_every_nth` and `preprocessing.sort_reverse` select the frames (a file that
/// is the output file itself is left out). The output format comes from the extension of the
/// output file, the bit depth, the JPEG quality and `copy_metadata` from
/// `settings.image_saving`. The out-of-core flow keeps one aligned frame and one padded tile
/// of every frame in memory; with `tile_size = 0` the whole stack is decoded into RAM.
///
/// # Errors
/// [`PipelineError::Cancelled`] when the hooks' cancel fires; [`PipelineError::Invalid`] for
/// an output file whose extension names no format, no frame left, a tile size below 64,
/// frames of different sizes; the decoder's, the alignment's, the tile store's and the
/// encoder's errors.
pub fn run_files(
    job: &FileJob<'_>,
    settings: &StackingSettings,
    hooks: &Hooks<'_>,
) -> Result<(), PipelineError> {
    let format = output_format_for_path(job.output_file).ok_or_else(|| {
        PipelineError::invalid(format!(
            "cannot tell the output format from {:?}; the extension must be one of .tif, .tiff, \
             .png, .jpg, .jpeg",
            job.output_file.display().to_string()
        ))
    })?;
    let output_path = job.output_file.to_path_buf();
    let paths: Vec<PathBuf> = select_frames(job.paths.to_vec(), settings)
        .into_iter()
        .filter(|path| !is_excluded(path, std::slice::from_ref(&output_path)))
        .collect();
    if paths.is_empty() {
        return Err(PipelineError::invalid(
            "No images to process: add some images first.",
        ));
    }
    if let Some(parent) = job.output_file.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let saving = ImageSavingSettings {
        output_format: format,
        ..settings.image_saving.clone()
    };
    if saving.copy_metadata && !supports_exif(format) {
        hooks
            .log
            .warn("A TIFF result cannot carry EXIF metadata; it is written without.");
    }
    let exif = source_exif(&paths[0], &settings.preprocessing, &saving);

    let tile_size = effective_tile_size(job.tile_size, settings);
    let mut source = PathSource::new(paths);
    let image = if tile_size > 0 {
        let scratch_root = std::env::temp_dir();
        let tiled = TiledJob {
            method: job.mode.method(),
            settings,
            tile_size,
            scratch_root: &scratch_root,
        };
        let mut compute = Compute::for_settings(settings);
        let bit16 = wants_16_bit(format, saving.bit_depth);
        let quality = jpeg_quality(saving.jpeg_quality);
        let TiledOutput { sink, crop, .. } = run_tiled(
            &mut source,
            &tiled,
            &mut compute,
            |width, height| EncodedImage::zeroed(width, height, format, bit16, quality),
            hooks,
        )?;
        match crop {
            Some(rect) => sink.crop(rect),
            None => sink,
        }
    } else {
        let loaded = load_frames(
            &mut source,
            &settings.preprocessing,
            hooks.cancel,
            hooks.progress,
        )?;
        if let Some(warning) = loaded.skipped_warning() {
            hooks.log.warn(&warning);
        }
        let fused = stack_in_ram(loaded.frames, settings, job.mode, hooks)?;
        encode(&fused, &saving)
    };

    hooks.progress.stage(Stage::Save, 0.0, "");
    image.with_exif(exif).save(job.output_file)?;
    hooks.progress.stage(Stage::Save, 1.0, "");
    Ok(())
}

// ── batches ─────────────────────────────────────────────────────────────────

/// The result of one sub-folder of a batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchEntry {
    /// The name of the sub-folder.
    pub name: String,
    /// The result file, or the error message of the stack that failed.
    pub result: Result<PathBuf, String>,
}

/// The direct sub-folders of `input_dir` that hold at least one frame, in natural name order.
///
/// # Errors
/// The I/O error when `input_dir` cannot be read.
pub fn batch_subfolders(input_dir: &Path) -> Result<Vec<PathBuf>, PipelineError> {
    let mut folders: Vec<PathBuf> = std::fs::read_dir(input_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && collect_image_paths(path, &[]).is_ok())
        .collect();
    natural_sort(&mut folders);
    Ok(folders)
}

/// The result file of the stack in the folder `name`: `filename_template` with `{name}`
/// replaced by the folder name, and the extension of the output format.
#[must_use]
#[allow(clippy::literal_string_with_formatting_args)]
pub fn batch_output_path(output_dir: &Path, name: &str, saving: &ImageSavingSettings) -> PathBuf {
    let stem = saving.filename_template.replace("{name}", name);
    output_dir.join(format!("{stem}.{}", extension_of(saving.output_format)))
}

/// Stack one sub-folder of a batch.
fn run_folder(
    folder: &Path,
    output_file: &Path,
    settings: &StackingSettings,
    mode: FuseMode,
    tile_size: usize,
    hooks: &Hooks<'_>,
) -> Result<(), PipelineError> {
    let paths = collect_image_paths(folder, &[])?;
    run_files(
        &FileJob {
            paths: &paths,
            output_file,
            mode,
            tile_size,
        },
        settings,
        hooks,
    )
}

/// Stack every direct sub-folder of `input_dir` that holds frames into its own file in
/// `output_dir` (see [`batch_output_path`]), one after the other.
///
/// A folder whose stack fails is recorded in the result and the batch goes on.
///
/// # Errors
/// [`PipelineError::Cancelled`] when the hooks' cancel fires (the batch stops); the I/O error
/// when `input_dir` cannot be read or `output_dir` cannot be created.
pub fn run_batch(
    input_dir: &Path,
    output_dir: &Path,
    settings: &StackingSettings,
    mode: FuseMode,
    tile_size: usize,
    hooks: &Hooks<'_>,
) -> Result<Vec<BatchEntry>, PipelineError> {
    std::fs::create_dir_all(output_dir)?;
    let mut entries = Vec::new();
    for folder in batch_subfolders(input_dir)? {
        let name = folder.file_name().map_or_else(
            || "stack".to_owned(),
            |name| name.to_string_lossy().into_owned(),
        );
        let output_file = batch_output_path(output_dir, &name, &settings.image_saving);
        match run_folder(&folder, &output_file, settings, mode, tile_size, hooks) {
            Ok(()) => entries.push(BatchEntry {
                name,
                result: Ok(output_file),
            }),
            Err(PipelineError::Cancelled) => return Err(PipelineError::Cancelled),
            Err(error) => entries.push(BatchEntry {
                name,
                result: Err(error.to_string()),
            }),
        }
    }
    Ok(entries)
}

// ── one frame ───────────────────────────────────────────────────────────────

/// Read one frame as 16-bit RGB samples, as `(width, height, samples)`.
///
/// The samples are the sRGB codes of the file (an 8-bit file is scaled to the 16-bit range
/// exactly: code `v` becomes `v * 257`); the EXIF orientation of a JPEG is applied. A camera
/// RAW file is developed by the RAW decoder (builds with the `raw` feature only).
///
/// # Errors
/// [`PipelineError::Decode`] when the file cannot be read or decoded.
pub fn read_rgb16(path: &Path) -> Result<(usize, usize, Vec<u16>), PipelineError> {
    let decoded = if has_raw_extension(path) {
        stacker_core::io::decode_raw_file(path).map_err(|e| PipelineError::decode(path, e))?
    } else {
        let bytes = std::fs::read(path).map_err(|e| PipelineError::decode(path, e))?;
        let image = image::load_from_memory(&bytes).map_err(|e| PipelineError::decode(path, e))?;
        let head = &bytes[..bytes.len().min(EXIF_HEAD_BYTES)];
        match jpeg_exif_orientation(head) {
            Some(orientation) => apply_orientation(image, orientation),
            None => image,
        }
    };
    // The `image` crate widens 8 bits with `<< 8` (255 -> 65280); an exact `* 257` keeps white
    // at 65535, so 8-bit sources are widened here.
    let eight_bit = matches!(
        decoded.color(),
        image::ColorType::L8
            | image::ColorType::La8
            | image::ColorType::Rgb8
            | image::ColorType::Rgba8
    );
    let (width, height, samples) = if eight_bit {
        let rgb = decoded.to_rgb8();
        let (width, height) = rgb.dimensions();
        let samples = rgb
            .into_raw()
            .into_iter()
            .map(|v| u16::from(v) * 257)
            .collect();
        (width, height, samples)
    } else {
        let rgb = decoded.to_rgb16();
        let (width, height) = rgb.dimensions();
        (width, height, rgb.into_raw())
    };
    Ok((width as usize, height as usize, samples))
}
