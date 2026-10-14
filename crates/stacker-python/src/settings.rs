//! Python-visible mirrors of `stacker_core::settings` and of the parameters of one file run.
//!
//! # "Break on change"
//!
//! The conversions [`PyStackingSettings::into_core`] and [`PyStackingSettings::from_core`]
//! (and those of the two nested mirrors) destructure the core structs **exhaustively**, with no
//! `..Default::default()` and no `..rest`. When `StackingSettings` gains a field this crate
//! stops compiling until somebody decides how Python sees the new key. A binding that compiles
//! and quietly drops a setting that changes the output is the worse failure.
//!
//! All 39 keys of the settings file (27 top-level, 6 in `preprocessing`, 6 in `image_saving`)
//! are mirrored under the name they have in the file. Enums are strings (see
//! [`crate::options`]). Two conveniences from earlier versions stay as properties and as
//! constructor arguments: `relief_engine` (`"guided_filter"` or `"multigrid"`, for
//! `relief_use_multigrid`) and `use_gpu` (a bool, for `compute_device`).
//!
//! # Validation
//!
//! [`PyStackingSettings::into_core`] accepts only what the settings file documents: a value
//! outside its range is an error naming the key and the range (the loaders of the GUI and the
//! CLI clamp and warn instead, which suits a hand-edited file but would hide a mistake in
//! code). The ranges are the ones of `StackingSettings::sanitized`, so there is a single
//! definition.
//!
//! The structs are plain Rust without the `python` feature, so the mirror is tested without an
//! interpreter; with the feature they are `pyclass`es.

use stacker_core::settings::{
    ComputeDevice, ImageSavingSettings, PreprocessingSettings, StackingSettings,
};

use crate::options::{
    ArgError, parse_alignment_mode, parse_compute_device, parse_optimizer, parse_output_format,
};

// ── preprocessing ───────────────────────────────────────────────────────────

/// Python mirror of `stacker_core::settings::PreprocessingSettings` (what is done to every
/// frame while it is loaded).
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all, from_py_object))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyPreprocessingSettings {
    /// Clockwise rotation in degrees: 0, 90, 180 or 270.
    pub pre_rotation: u32,
    /// Whether the crop in `pre_crop_spec` is applied.
    pub pre_crop_enabled: bool,
    /// Crop as `"w,h"` (centred) or `"x,y,w,h"`.
    pub pre_crop_spec: String,
    /// Resize every frame to this percentage of its size, `10..=100`.
    pub pre_resize_percent: u32,
    /// Process the frames in reverse order.
    pub sort_reverse: bool,
    /// Ignore the EXIF orientation tag when decoding.
    pub ignore_exif_orientation: bool,
}

impl Default for PyPreprocessingSettings {
    fn default() -> Self {
        Self::from_core(&PreprocessingSettings::default())
    }
}

impl PyPreprocessingSettings {
    /// The core settings. Exhaustively destructured, see the module documentation.
    #[must_use]
    pub fn into_core(self) -> PreprocessingSettings {
        let Self {
            pre_rotation,
            pre_crop_enabled,
            pre_crop_spec,
            pre_resize_percent,
            sort_reverse,
            ignore_exif_orientation,
        } = self;
        PreprocessingSettings {
            pre_rotation,
            pre_crop_enabled,
            pre_crop_spec,
            pre_resize_percent,
            sort_reverse,
            ignore_exif_orientation,
        }
    }

    /// The mirror of core settings. Exhaustively destructured, see the module documentation.
    #[must_use]
    pub fn from_core(core: &PreprocessingSettings) -> Self {
        let PreprocessingSettings {
            pre_rotation,
            pre_crop_enabled,
            pre_crop_spec,
            pre_resize_percent,
            sort_reverse,
            ignore_exif_orientation,
        } = core.clone();
        Self {
            pre_rotation,
            pre_crop_enabled,
            pre_crop_spec,
            pre_resize_percent,
            sort_reverse,
            ignore_exif_orientation,
        }
    }
}

// ── image saving ────────────────────────────────────────────────────────────

/// Python mirror of `stacker_core::settings::ImageSavingSettings` (how the result is written).
///
/// `output_format` is `"TIFF"`, `"PNG"` or `"JPEG"` (any letter case; `"tif"` and `"jpg"` work
/// too). `stack_files` takes the format from the extension of its `output_file`; `batch_stack`
/// takes it from here.
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all, from_py_object))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyImageSavingSettings {
    /// `"TIFF"`, `"PNG"` or `"JPEG"`.
    pub output_format: String,
    /// Bits per sample, 8 or 16 (JPEG is always 8).
    pub bit_depth: u32,
    /// JPEG quality, `1..=100`.
    pub jpeg_quality: u32,
    /// Output file name without extension; `{name}` is the name of the stack's folder.
    pub filename_template: String,
    /// Directory results go to (not used by `stack_files`, which takes a path).
    pub default_output_dir: String,
    /// Copy the EXIF block of the first frame into the result (JPEG and PNG; a TIFF result
    /// cannot carry it and a warning says so).
    pub copy_metadata: bool,
}

impl Default for PyImageSavingSettings {
    fn default() -> Self {
        Self::from_core(&ImageSavingSettings::default())
    }
}

impl PyImageSavingSettings {
    /// The core settings.
    ///
    /// # Errors
    /// [`ArgError`] when `output_format` is not one of the valid labels.
    pub fn into_core(self) -> Result<ImageSavingSettings, ArgError> {
        let Self {
            output_format,
            bit_depth,
            jpeg_quality,
            filename_template,
            default_output_dir,
            copy_metadata,
        } = self;
        Ok(ImageSavingSettings {
            output_format: parse_output_format(&output_format)?,
            bit_depth,
            jpeg_quality,
            filename_template,
            default_output_dir,
            copy_metadata,
        })
    }

    /// The mirror of core settings. Exhaustively destructured, see the module documentation.
    #[must_use]
    pub fn from_core(core: &ImageSavingSettings) -> Self {
        let ImageSavingSettings {
            output_format,
            bit_depth,
            jpeg_quality,
            filename_template,
            default_output_dir,
            copy_metadata,
        } = core.clone();
        Self {
            output_format: output_format.as_combo_str().to_owned(),
            bit_depth,
            jpeg_quality,
            filename_template,
            default_output_dir,
            copy_metadata,
        }
    }
}

// ── stacking settings ───────────────────────────────────────────────────────

/// Python mirror of `stacker_core::settings::StackingSettings`: the option set of one run, the
/// same keys the settings file, the GUI and the CLI use.
///
/// Every field has the name of its key in the settings file. The enum fields are strings
/// (`alignment_mode`: `Translation`, `Registration`, `Affine`, `None`, `Neural`; `optimizer`:
/// `Auto`, `Lucas-Kanade`, `Nelder-Mead`; `compute_device`: `CPU`, `GPU`). An invalid string or
/// a number outside its documented range raises `ValueError` when the settings are used.
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all, from_py_object))]
#[derive(Clone, Debug, PartialEq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent toggles, mirrored one-to-one from the settings file"
)]
pub struct PyStackingSettings {
    /// Schema version of the settings file; leave it at the default.
    pub settings_version: u32,
    /// Registration model: `"Translation"`, `"Registration"`, `"Affine"`, `"None"` or
    /// `"Neural"` (without a neural model the pipeline falls back to `Registration`).
    pub alignment_mode: String,
    /// Intensity optimiser of the classical alignment modes: `"Auto"`, `"Lucas-Kanade"` or
    /// `"Nelder-Mead"`.
    pub optimizer: String,
    /// Constrain the optimiser to a bounded band around the initial estimate.
    pub bounded_search: bool,
    /// Seed the optimiser with an AKAZE feature match (needs a build with `akaze`).
    pub akaze_seeding: bool,
    /// Match the brightness of every frame to the reference (in linear light).
    pub correct_brightness: bool,
    /// Align each frame to its aligned predecessor instead of to the reference.
    pub chain_alignment: bool,
    /// With `alignment_mode = "Neural"`: refine the network's matrices classically.
    pub neural_refine_classically: bool,
    /// Crop the result to the area every aligned frame covers.
    pub crop_to_common_area: bool,
    /// After that crop, resample the result back to the original size.
    pub resize_cropped_to_original: bool,
    /// Drop frames that win too little of the detail before fusing.
    pub auto_cull: bool,
    /// Percentage of the detail a frame must win to survive the cull, `0.0..=50.0`; `0.0` is
    /// the automatic rule that scales with the frame count.
    pub auto_cull_threshold_pct: f32,
    /// Fuse the frames sharpest-first (nearest-neighbour order of the focus census).
    pub sort_by_sharpness: bool,
    /// Use only every N-th frame (`>= 1`).
    pub stack_every_nth: u32,
    /// Edge of the out-of-core tiles in pixels: `0` or `64..=8192`. `stack_files` uses its
    /// own `tile_size` argument first and this value when that is `0`; `0` in both means the
    /// stack runs in RAM.
    pub tile_size: u32,
    /// `Apex`: pyramid depth, `2..=12`.
    pub pyramid_levels: u32,
    /// `Apex`: include the chroma planes in the selection energy.
    pub use_all_color_channels: bool,
    /// `Apex`: choose the winner at the finest band by the 3x3 neighbourhood energy.
    pub grit_suppression: bool,
    /// `Relief`: window radius of the focus measure, `0..=100`.
    pub relief_estimation_radius: u32,
    /// `Relief`: guided-filter radius of the depth-map smoothing, `0..=50`.
    pub relief_smoothing_radius: u32,
    /// `Relief`: fraction of the pixels treated as no signal, `0.0..=1.0`.
    pub relief_contrast_pct: f32,
    /// `Relief`: GUI preview of the contrast mask (no effect from Python; kept so a file
    /// round-trips).
    pub relief_show_preview: bool,
    /// `Relief`: choose the contrast threshold automatically.
    pub relief_auto_detect: bool,
    /// `Relief`: solve the depth map with the multigrid solver (also `relief_engine`).
    pub relief_use_multigrid: bool,
    /// `Strata`: box radius that splits base and detail, `8..=64`.
    pub strata_base_radius: u32,
    /// `Strata`: how strongly the detail weights concentrate on the sharpest frame, `1..=5`.
    pub strata_detail_focus: u32,
    /// Where the heavy stages run: `"CPU"` or `"GPU"` (also `use_gpu`; needs a build with
    /// `gpu`, otherwise the CPU is used and a warning says so).
    pub compute_device: String,
    /// What is done to every frame while it is loaded.
    pub preprocessing: PyPreprocessingSettings,
    /// How the result is written.
    pub image_saving: PyImageSavingSettings,
}

impl Default for PyStackingSettings {
    fn default() -> Self {
        Self::from_core(&StackingSettings::default())
    }
}

impl PyStackingSettings {
    /// The core settings, checked.
    ///
    /// Exhaustively destructured with no `..Default::default()`, see the module
    /// documentation. Besides the enum strings, every number must be inside the range the
    /// settings file documents.
    ///
    /// # Errors
    /// [`ArgError`] for an unknown option string, or listing every key that is outside its
    /// range.
    pub fn into_core(self) -> Result<StackingSettings, ArgError> {
        let Self {
            settings_version,
            alignment_mode,
            optimizer,
            bounded_search,
            akaze_seeding,
            correct_brightness,
            chain_alignment,
            neural_refine_classically,
            crop_to_common_area,
            resize_cropped_to_original,
            auto_cull,
            auto_cull_threshold_pct,
            sort_by_sharpness,
            stack_every_nth,
            tile_size,
            pyramid_levels,
            use_all_color_channels,
            grit_suppression,
            relief_estimation_radius,
            relief_smoothing_radius,
            relief_contrast_pct,
            relief_show_preview,
            relief_auto_detect,
            relief_use_multigrid,
            strata_base_radius,
            strata_detail_focus,
            compute_device,
            preprocessing,
            image_saving,
        } = self;

        let (alignment_mode, bounded_by_label) = parse_alignment_mode(&alignment_mode)?;
        let core = StackingSettings {
            settings_version,
            alignment_mode,
            optimizer: parse_optimizer(&optimizer)?,
            bounded_search: bounded_search || bounded_by_label,
            akaze_seeding,
            correct_brightness,
            chain_alignment,
            neural_refine_classically,
            crop_to_common_area,
            resize_cropped_to_original,
            auto_cull,
            auto_cull_threshold_pct,
            sort_by_sharpness,
            stack_every_nth,
            tile_size,
            pyramid_levels,
            use_all_color_channels,
            grit_suppression,
            relief_estimation_radius,
            relief_smoothing_radius,
            relief_contrast_pct,
            relief_show_preview,
            relief_auto_detect,
            relief_use_multigrid,
            strata_base_radius,
            strata_detail_focus,
            compute_device: parse_compute_device(&compute_device)?,
            preprocessing: preprocessing.into_core(),
            image_saving: image_saving.into_core()?,
        };
        let (clean, problems) = core.sanitized();
        if problems.is_empty() {
            Ok(clean)
        } else {
            Err(ArgError::new(format!(
                "invalid settings: {}",
                problems.join("; ")
            )))
        }
    }

    /// The mirror of core settings.
    ///
    /// Exhaustively destructured with no `..Default::default()`, see the module
    /// documentation.
    #[must_use]
    pub fn from_core(core: &StackingSettings) -> Self {
        let StackingSettings {
            settings_version,
            alignment_mode,
            optimizer,
            bounded_search,
            akaze_seeding,
            correct_brightness,
            chain_alignment,
            neural_refine_classically,
            crop_to_common_area,
            resize_cropped_to_original,
            auto_cull,
            auto_cull_threshold_pct,
            sort_by_sharpness,
            stack_every_nth,
            tile_size,
            pyramid_levels,
            use_all_color_channels,
            grit_suppression,
            relief_estimation_radius,
            relief_smoothing_radius,
            relief_contrast_pct,
            relief_show_preview,
            relief_auto_detect,
            relief_use_multigrid,
            strata_base_radius,
            strata_detail_focus,
            compute_device,
            preprocessing,
            image_saving,
        } = core.clone();
        Self {
            settings_version,
            alignment_mode: alignment_mode.as_combo_str().to_owned(),
            optimizer: optimizer.as_combo_str().to_owned(),
            bounded_search,
            akaze_seeding,
            correct_brightness,
            chain_alignment,
            neural_refine_classically,
            crop_to_common_area,
            resize_cropped_to_original,
            auto_cull,
            auto_cull_threshold_pct,
            sort_by_sharpness,
            stack_every_nth,
            tile_size,
            pyramid_levels,
            use_all_color_channels,
            grit_suppression,
            relief_estimation_radius,
            relief_smoothing_radius,
            relief_contrast_pct,
            relief_show_preview,
            relief_auto_detect,
            relief_use_multigrid,
            strata_base_radius,
            strata_detail_focus,
            compute_device: compute_device.as_combo_str().to_owned(),
            preprocessing: PyPreprocessingSettings::from_core(&preprocessing),
            image_saving: PyImageSavingSettings::from_core(&image_saving),
        }
    }

    /// The `compute_device` label of a `use_gpu` flag.
    #[must_use]
    pub const fn device_label(use_gpu: bool) -> &'static str {
        if use_gpu {
            ComputeDevice::Gpu.as_combo_str()
        } else {
            ComputeDevice::Cpu.as_combo_str()
        }
    }

    /// `true` when `compute_device` asks for the GPU (an unreadable label counts as no).
    #[must_use]
    pub fn uses_gpu(&self) -> bool {
        parse_compute_device(&self.compute_device).is_ok_and(|device| device == ComputeDevice::Gpu)
    }
}

// ── one file run ────────────────────────────────────────────────────────────

/// Python class: the parameters of one `stack_files` run.
///
/// `mode` is `"apex"`, `"relief"` or `"strata"` (the older names `"pmax"` and `"dmap"` still
/// work). `tile_size` is the edge of the out-of-core tiles; with `0` (and `tile_size = 0` in
/// the settings) the frames are held in RAM instead.
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all, from_py_object))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyPipelineParams {
    /// The input frames, in stack order (the first is the alignment reference).
    pub paths: Vec<String>,
    /// Where the result goes; the extension (`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`) picks
    /// the format.
    pub output_file: String,
    /// The fusion method.
    pub mode: String,
    /// Edge of the out-of-core tiles in pixels (`>= 64`), or `0`.
    pub tile_size: usize,
}

impl PyPipelineParams {
    /// Parameters for [`crate::file_api::stack_files`].
    #[must_use]
    pub const fn new(
        paths: Vec<String>,
        output_file: String,
        mode: String,
        tile_size: usize,
    ) -> Self {
        Self {
            paths,
            output_file,
            mode,
            tile_size,
        }
    }
}

// ── methods Python sees ─────────────────────────────────────────────────────

#[cfg(feature = "python")]
mod python {
    use pyo3::{PyResult, pymethods};

    use super::{
        PyImageSavingSettings, PyPipelineParams, PyPreprocessingSettings, PyStackingSettings,
    };
    use crate::options::{parse_relief_engine, relief_engine_str};

    #[pymethods]
    impl PyPreprocessingSettings {
        #[new]
        #[pyo3(signature = (
            pre_rotation=0,
            pre_crop_enabled=false,
            pre_crop_spec=String::new(),
            pre_resize_percent=100,
            sort_reverse=false,
            ignore_exif_orientation=false,
        ))]
        const fn py_new(
            pre_rotation: u32,
            pre_crop_enabled: bool,
            pre_crop_spec: String,
            pre_resize_percent: u32,
            sort_reverse: bool,
            ignore_exif_orientation: bool,
        ) -> Self {
            Self {
                pre_rotation,
                pre_crop_enabled,
                pre_crop_spec,
                pre_resize_percent,
                sort_reverse,
                ignore_exif_orientation,
            }
        }

        fn __eq__(&self, other: &Self) -> bool {
            self == other
        }

        fn __repr__(&self) -> String {
            format!(
                "PreprocessingSettings(pre_rotation={}, pre_crop_enabled={}, \
                 pre_resize_percent={}, sort_reverse={})",
                self.pre_rotation,
                self.pre_crop_enabled,
                self.pre_resize_percent,
                self.sort_reverse
            )
        }
    }

    #[pymethods]
    impl PyImageSavingSettings {
        #[new]
        #[pyo3(signature = (
            output_format="TIFF".to_string(),
            bit_depth=16,
            jpeg_quality=95,
            filename_template="{name}_stacked".to_string(),
            default_output_dir=String::new(),
            copy_metadata=false,
        ))]
        const fn py_new(
            output_format: String,
            bit_depth: u32,
            jpeg_quality: u32,
            filename_template: String,
            default_output_dir: String,
            copy_metadata: bool,
        ) -> Self {
            Self {
                output_format,
                bit_depth,
                jpeg_quality,
                filename_template,
                default_output_dir,
                copy_metadata,
            }
        }

        fn __eq__(&self, other: &Self) -> bool {
            self == other
        }

        fn __repr__(&self) -> String {
            format!(
                "ImageSavingSettings(output_format={:?}, bit_depth={}, jpeg_quality={})",
                self.output_format, self.bit_depth, self.jpeg_quality
            )
        }
    }

    #[pymethods]
    impl PyPipelineParams {
        #[new]
        #[pyo3(signature = (paths, output_file, mode="apex".to_string(), tile_size=512))]
        const fn py_new(
            paths: Vec<String>,
            output_file: String,
            mode: String,
            tile_size: usize,
        ) -> Self {
            Self::new(paths, output_file, mode, tile_size)
        }

        fn __eq__(&self, other: &Self) -> bool {
            self == other
        }

        fn __repr__(&self) -> String {
            format!(
                "PipelineParams(paths=[{} frames], output_file={:?}, mode={:?}, tile_size={})",
                self.paths.len(),
                self.output_file,
                self.mode,
                self.tile_size
            )
        }
    }

    #[pymethods]
    impl PyStackingSettings {
        /// Settings with the defaults of `StackingSettings::default()`; every argument that is
        /// not given keeps its default.
        #[new]
        #[pyo3(signature = (
            settings_version=None,
            alignment_mode=None,
            optimizer=None,
            bounded_search=None,
            akaze_seeding=None,
            correct_brightness=None,
            chain_alignment=None,
            neural_refine_classically=None,
            crop_to_common_area=None,
            resize_cropped_to_original=None,
            auto_cull=None,
            auto_cull_threshold_pct=None,
            sort_by_sharpness=None,
            stack_every_nth=None,
            tile_size=None,
            pyramid_levels=None,
            use_all_color_channels=None,
            grit_suppression=None,
            relief_estimation_radius=None,
            relief_smoothing_radius=None,
            relief_contrast_pct=None,
            relief_show_preview=None,
            relief_auto_detect=None,
            relief_use_multigrid=None,
            strata_base_radius=None,
            strata_detail_focus=None,
            compute_device=None,
            preprocessing=None,
            image_saving=None,
            relief_engine=None,
            use_gpu=None,
        ))]
        #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
        fn py_new(
            settings_version: Option<u32>,
            alignment_mode: Option<String>,
            optimizer: Option<String>,
            bounded_search: Option<bool>,
            akaze_seeding: Option<bool>,
            correct_brightness: Option<bool>,
            chain_alignment: Option<bool>,
            neural_refine_classically: Option<bool>,
            crop_to_common_area: Option<bool>,
            resize_cropped_to_original: Option<bool>,
            auto_cull: Option<bool>,
            auto_cull_threshold_pct: Option<f32>,
            sort_by_sharpness: Option<bool>,
            stack_every_nth: Option<u32>,
            tile_size: Option<u32>,
            pyramid_levels: Option<u32>,
            use_all_color_channels: Option<bool>,
            grit_suppression: Option<bool>,
            relief_estimation_radius: Option<u32>,
            relief_smoothing_radius: Option<u32>,
            relief_contrast_pct: Option<f32>,
            relief_show_preview: Option<bool>,
            relief_auto_detect: Option<bool>,
            relief_use_multigrid: Option<bool>,
            strata_base_radius: Option<u32>,
            strata_detail_focus: Option<u32>,
            compute_device: Option<String>,
            preprocessing: Option<PyPreprocessingSettings>,
            image_saving: Option<PyImageSavingSettings>,
            relief_engine: Option<String>,
            use_gpu: Option<bool>,
        ) -> PyResult<Self> {
            let d = Self::default();
            // The symbolic spellings win over the keys they stand for.
            let relief_use_multigrid = match relief_engine {
                Some(engine) => parse_relief_engine(&engine)?,
                None => relief_use_multigrid.unwrap_or(d.relief_use_multigrid),
            };
            let compute_device = match use_gpu {
                Some(gpu) => Self::device_label(gpu).to_owned(),
                None => compute_device.unwrap_or(d.compute_device),
            };
            Ok(Self {
                settings_version: settings_version.unwrap_or(d.settings_version),
                alignment_mode: alignment_mode.unwrap_or(d.alignment_mode),
                optimizer: optimizer.unwrap_or(d.optimizer),
                bounded_search: bounded_search.unwrap_or(d.bounded_search),
                akaze_seeding: akaze_seeding.unwrap_or(d.akaze_seeding),
                correct_brightness: correct_brightness.unwrap_or(d.correct_brightness),
                chain_alignment: chain_alignment.unwrap_or(d.chain_alignment),
                neural_refine_classically: neural_refine_classically
                    .unwrap_or(d.neural_refine_classically),
                crop_to_common_area: crop_to_common_area.unwrap_or(d.crop_to_common_area),
                resize_cropped_to_original: resize_cropped_to_original
                    .unwrap_or(d.resize_cropped_to_original),
                auto_cull: auto_cull.unwrap_or(d.auto_cull),
                auto_cull_threshold_pct: auto_cull_threshold_pct
                    .unwrap_or(d.auto_cull_threshold_pct),
                sort_by_sharpness: sort_by_sharpness.unwrap_or(d.sort_by_sharpness),
                stack_every_nth: stack_every_nth.unwrap_or(d.stack_every_nth),
                tile_size: tile_size.unwrap_or(d.tile_size),
                pyramid_levels: pyramid_levels.unwrap_or(d.pyramid_levels),
                use_all_color_channels: use_all_color_channels.unwrap_or(d.use_all_color_channels),
                grit_suppression: grit_suppression.unwrap_or(d.grit_suppression),
                relief_estimation_radius: relief_estimation_radius
                    .unwrap_or(d.relief_estimation_radius),
                relief_smoothing_radius: relief_smoothing_radius
                    .unwrap_or(d.relief_smoothing_radius),
                relief_contrast_pct: relief_contrast_pct.unwrap_or(d.relief_contrast_pct),
                relief_show_preview: relief_show_preview.unwrap_or(d.relief_show_preview),
                relief_auto_detect: relief_auto_detect.unwrap_or(d.relief_auto_detect),
                relief_use_multigrid,
                strata_base_radius: strata_base_radius.unwrap_or(d.strata_base_radius),
                strata_detail_focus: strata_detail_focus.unwrap_or(d.strata_detail_focus),
                compute_device,
                preprocessing: preprocessing.unwrap_or(d.preprocessing),
                image_saving: image_saving.unwrap_or(d.image_saving),
            })
        }

        /// `"guided_filter"` or `"multigrid"`: the `Relief` depth-map engine
        /// (`relief_use_multigrid` as a label).
        #[getter(relief_engine)]
        const fn engine(&self) -> &'static str {
            relief_engine_str(self.relief_use_multigrid)
        }

        #[setter(relief_engine)]
        fn set_engine(&mut self, value: String) -> PyResult<()> {
            self.relief_use_multigrid = parse_relief_engine(&value)?;
            Ok(())
        }

        /// `True` when `compute_device` is `"GPU"` (`compute_device` as a flag).
        #[getter(use_gpu)]
        fn gpu_flag(&self) -> bool {
            self.uses_gpu()
        }

        #[setter(use_gpu)]
        fn set_gpu_flag(&mut self, value: bool) {
            Self::device_label(value).clone_into(&mut self.compute_device);
        }

        fn __eq__(&self, other: &Self) -> bool {
            self == other
        }

        fn __repr__(&self) -> String {
            format!(
                "StackingSettings(alignment_mode={:?}, optimizer={:?}, auto_cull={}, \
                 crop_to_common_area={}, relief_use_multigrid={}, compute_device={:?})",
                self.alignment_mode,
                self.optimizer,
                self.auto_cull,
                self.crop_to_common_area,
                self.relief_use_multigrid,
                self.compute_device
            )
        }
    }
}
