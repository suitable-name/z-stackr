use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AlignmentModeSetting {
    #[default]
    Affine,
    Translation,
    Registration,
    None,
    #[cfg(feature = "nn")]
    Neural,
}

impl AlignmentModeSetting {
    #[must_use]
    pub const fn as_combo_str(self) -> &'static str {
        match self {
            Self::Affine => "Affine",
            Self::Translation => "Translation",
            Self::Registration => "Registration",
            Self::None => "None",
            #[cfg(feature = "nn")]
            Self::Neural => "Neural",
        }
    }
    #[must_use]
    pub fn from_combo_str(s: &str) -> Self {
        match s {
            "Translation" => Self::Translation,
            "Registration" => Self::Registration,
            "None" => Self::None,
            #[cfg(feature = "nn")]
            "Neural" => Self::Neural,
            _ => Self::Affine,
        }
    }
}

/// Selects which intensity-based optimiser drives subpixel alignment
/// refinement (see `stacker_align::refine`'s module docs for the algorithms
/// themselves).
///
/// - [`Self::Auto`] (default): try the pyramid inverse-compositional
///   Lucas-Kanade / Gauss-Newton optimiser first (`refine_alignment_lk`,
///   much cheaper per iteration than Nelder-Mead); if it errors or its final
///   RMS is not better than its own starting RMS, fall back to the Nelder-Mead
///   bounded simplex (`refine_alignment_registration`) from the same
///   sanity-filtered seed.
/// - [`Self::LucasKanade`]: always use the Lucas-Kanade optimiser. On error
///   or non-finite output there is **no** Nelder-Mead fallback — only a
///   fallback to the sanity-filtered initial matrix (same graceful
///   degradation as a Nelder-Mead `Err` has always had).
/// - [`Self::NelderMead`]: always use the Nelder-Mead bounded simplex —
///   exactly today's (pre-Lucas-Kanade) behaviour.
///
/// Every variant still passes through the same post-refinement
/// `refined_beats_identity` sanity gate before being accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerSetting {
    #[default]
    Auto,
    LucasKanade,
    NelderMead,
}

impl OptimizerSetting {
    #[must_use]
    pub const fn as_combo_str(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::LucasKanade => "Lucas-Kanade",
            Self::NelderMead => "Nelder-Mead",
        }
    }
    #[must_use]
    pub fn from_combo_str(s: &str) -> Self {
        match s {
            "Lucas-Kanade" => Self::LucasKanade,
            "Nelder-Mead" => Self::NelderMead,
            _ => Self::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Tiff,
    Png,
    Jpeg,
}

impl OutputFormat {
    #[must_use]
    pub const fn as_combo_str(self) -> &'static str {
        match self {
            Self::Tiff => "TIFF",
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
        }
    }
    #[must_use]
    pub fn from_combo_str(s: &str) -> Self {
        match s {
            "PNG" => Self::Png,
            "JPEG" => Self::Jpeg,
            _ => Self::Tiff,
        }
    }

    /// The file extension (no leading dot) written by this format.
    ///
    /// Single source of truth for "what extension does this output format
    /// use" — mirrors the mapping the GUI's Save dialog has inlined
    /// (`apps/stacker-gui/src/main.rs`'s `perform_save_current_image`) and
    /// is also used by the CLI's subfolder-batch mode
    /// (`apps/stacker-cli/src/batch.rs`) to derive each stack's output
    /// filename from `ImageSavingSettings` without duplicating the mapping
    /// a third time.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Tiff => "tiff",
            Self::Png => "png",
            Self::Jpeg => "jpg",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PreprocessingSettings {
    pub pre_rotation: u32,
    pub pre_crop_enabled: bool,
    pub pre_crop_spec: String,
    pub pre_resize_percent: u32,
    pub sort_reverse: bool,
    pub ignore_exif_orientation: bool,
}

impl Default for PreprocessingSettings {
    fn default() -> Self {
        Self {
            pre_rotation: 0,
            pre_crop_enabled: false,
            pre_crop_spec: String::new(),
            pre_resize_percent: 100,
            sort_reverse: false,
            ignore_exif_orientation: false,
        }
    }
}

impl PreprocessingSettings {
    pub fn clamp_valid(&mut self) {
        if !matches!(self.pre_rotation, 0 | 90 | 180 | 270) {
            self.pre_rotation = 0;
        }
        self.pre_resize_percent = self.pre_resize_percent.clamp(10, 100);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageSavingSettings {
    pub output_format: OutputFormat,
    pub bit_depth: u32,
    pub jpeg_quality: u32,
    pub filename_template: String,
    pub default_output_dir: String,
    pub copy_metadata: bool,
}

impl Default for ImageSavingSettings {
    fn default() -> Self {
        Self {
            output_format: OutputFormat::Tiff,
            bit_depth: 16,
            jpeg_quality: 95,
            filename_template: "{name}_stacked".to_owned(),
            default_output_dir: String::new(),
            copy_metadata: false,
        }
    }
}

impl ImageSavingSettings {
    pub fn clamp_valid(&mut self) {
        if !matches!(self.bit_depth, 8 | 16) {
            self.bit_depth = 16;
        }
        self.jpeg_quality = self.jpeg_quality.clamp(1, 100);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AlignmentFlags {
    pub akaze_seeding: bool,
    pub neural_refine_classically: bool,
    pub correct_brightness: bool,
}

impl Default for AlignmentFlags {
    fn default() -> Self {
        Self {
            akaze_seeding: false,
            neural_refine_classically: true,
            correct_brightness: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CullFlags {
    pub auto_cull: bool,
    pub sort_by_sharpness: bool,
}

impl Default for CullFlags {
    fn default() -> Self {
        Self {
            auto_cull: true,
            sort_by_sharpness: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ApexFlags {
    pub use_all_color_channels: bool,
    pub grit_suppression: bool,
}

impl Default for ApexFlags {
    fn default() -> Self {
        Self {
            use_all_color_channels: false,
            grit_suppression: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct ReliefFlags {
    pub relief_show_preview: bool,
    pub relief_auto_detect: bool,
    pub relief_use_multigrid: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CropFlags {
    pub crop_to_common_area: bool,
    pub resize_cropped_to_original: bool,
}

impl Default for CropFlags {
    fn default() -> Self {
        Self {
            crop_to_common_area: true,
            resize_cropped_to_original: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuFlags {
    pub use_gpu: bool,
}

impl Default for GpuFlags {
    fn default() -> Self {
        Self { use_gpu: true }
    }
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StackingSettings {
    pub alignment_mode: AlignmentModeSetting,
    pub optimizer: OptimizerSetting,
    pub akaze_seeding: bool,
    pub neural_refine_classically: bool,
    pub correct_brightness: bool,
    pub auto_cull: bool,
    pub sort_by_sharpness: bool,
    pub tile_size: u32,
    pub stack_every_nth: u32,
    pub pyramid_levels: u32,
    pub use_all_color_channels: bool,
    pub grit_suppression: bool,
    pub relief_estimation_radius: u32,
    pub relief_smoothing_radius: u32,
    pub relief_contrast_pct: f32,
    pub relief_show_preview: bool,
    pub relief_auto_detect: bool,
    pub relief_use_multigrid: bool,
    pub strata_base_radius: u32,
    pub strata_detail_focus: u32,
    pub crop_to_common_area: bool,
    pub resize_cropped_to_original: bool,
    pub auto_cull_threshold_pct: f32,
    pub use_gpu: bool,
    pub preprocessing: PreprocessingSettings,
    pub image_saving: ImageSavingSettings,
}

impl Default for StackingSettings {
    fn default() -> Self {
        Self {
            alignment_mode: AlignmentModeSetting::Registration,
            optimizer: OptimizerSetting::Auto,
            akaze_seeding: false,
            neural_refine_classically: true,
            correct_brightness: true,
            auto_cull: true,
            sort_by_sharpness: true,
            tile_size: 0,
            stack_every_nth: 1,
            pyramid_levels: 8,
            use_all_color_channels: false,
            grit_suppression: true,
            relief_estimation_radius: 5,
            relief_smoothing_radius: 2,
            relief_contrast_pct: 0.0,
            relief_show_preview: false,
            relief_auto_detect: false,
            relief_use_multigrid: false,
            strata_base_radius: 31,
            strata_detail_focus: 3,
            crop_to_common_area: true,
            resize_cropped_to_original: false,
            auto_cull_threshold_pct: 2.0,
            use_gpu: true,
            preprocessing: PreprocessingSettings::default(),
            image_saving: ImageSavingSettings::default(),
        }
    }
}

impl StackingSettings {
    /// Clamp the top-level fields that have a bounded valid range but no
    /// sub-struct of their own to fall back on. Does **not** call
    /// [`PreprocessingSettings::clamp_valid`] or
    /// [`ImageSavingSettings::clamp_valid`] — every existing call site
    /// already invokes those two explicitly, and duplicating them here
    /// would silently change behaviour at call sites that construct a
    /// `StackingSettings` without its sub-structs populated yet. Callers
    /// that deserialize a full `StackingSettings` (GUI settings load, GUI
    /// Load-Config dialog, CLI `--config`) should keep calling all three
    /// `clamp_valid` methods alongside each other, as before.
    pub const fn clamp_valid(&mut self) {
        self.auto_cull_threshold_pct = self.auto_cull_threshold_pct.clamp(0.1, 5.0);
        // `u32::clamp` (via `Ord`) is not yet usable in `const fn` — see
        // rust-lang/rust#143874 — so clamp manually here.
        if self.strata_base_radius < 8 {
            self.strata_base_radius = 8;
        } else if self.strata_base_radius > 64 {
            self.strata_base_radius = 64;
        }
        if self.strata_detail_focus < 1 {
            self.strata_detail_focus = 1;
        } else if self.strata_detail_focus > 5 {
            self.strata_detail_focus = 5;
        }
    }
}

pub const DEFAULT_CONFIG_TOML: &str = r#"# Focus Stacker RS - Configuration File
# This file is automatically generated. Any missing settings will use their defaults.

# ── Alignment ────────────────────────────────────────────────────────────
# Registration model applied before fusion. Every mode uses intensity-based
# registration (RMS-difference minimisation over a coarse-to-fine luma
# pyramid), always bounded to a logit-space search so the optimiser can
# never drift into a degenerate transform. The three active modes form a
# strict ladder, each solving a superset of the DOFs the previous one does:
# "translation" ⊂ "registration" (default) ⊂ "affine".
# Valid values:
#   "translation"  - shift (X / Y) + focus-breathing scale, no rotation
#   "registration" - shift + rotation + uniform scale (a similarity
#                    transform) via the full multi-scale coarse-to-fine
#                    bounded registration (edge-clamped spline interpolation,
#                    Gaussian pyramid and coarsest-level random restarts;
#                    finest levels are deliberately under-refined to avoid
#                    chasing per-frame noise). This is the default.
#   "affine"       - everything "registration" solves, plus separate X/Y
#                    scale (anisotropic scale) and shear: a true 6-DOF
#                    affine solve. Use when frames need independent X/Y
#                    scaling or shear correction that a similarity transform
#                    cannot represent.
#   "none"         - skip alignment entirely
alignment_mode = "registration"

# ── Optimizer ────────────────────────────────────────────────────────────
# Selects which intensity-based optimiser performs the subpixel refinement
# described above. Every classical alignment_mode ("translation",
# "registration", "affine") is refined by whichever optimiser this selects;
# the DOF gating from alignment_mode is unchanged either way.
# Valid values:
#   "auto"        - (default) try the pyramid inverse-compositional
#                    Lucas-Kanade / Gauss-Newton optimiser first (many fewer
#                    iterations than Nelder-Mead, since it uses analytic
#                    gradients instead of blind function sampling). If it
#                    fails to converge (errors, produces non-finite output,
#                    or its final RMS is not better than its own starting
#                    RMS) this automatically falls back to the Nelder-Mead
#                    bounded simplex from the same seed.
#   "lucaskanade" - always use the Lucas-Kanade optimiser. On failure there
#                    is NO Nelder-Mead fallback (only the same
#                    sanity-filtered-initial-matrix fallback Nelder-Mead
#                    itself has always had on error).
#   "neldermead"  - always use the Nelder-Mead bounded simplex,
#                    unconditionally.
# All three still pass through the same post-refinement RMS-vs-identity
# sanity gate before the result is accepted.
optimizer = "auto"

# Seed the optimiser with an AKAZE feature-match estimate before intensity
# refinement. Helps on textured subjects with large frame-to-frame shifts.
# Leave false for pure intensity registration, which is more robust on
# glossy / featureless subjects. Has no effect unless the application was
# built with the `akaze` feature.
akaze_seeding = false

# Apply per-frame brightness/gamma correction after warping.
correct_brightness = true

# Hybrid neural alignment: when alignment_mode = "neural" (requires the `nn`
# build feature), feed the network's matrix into the existing classical
# intensity optimiser as a seed (the same machinery an AKAZE seed feeds into)
# instead of applying it directly. The seed is still sanity-filtered and the
# refinement is still gated against ending up worse than identity, so this
# degrades gracefully to classical behaviour when the neural estimate is
# poor. false uses the network's matrix directly (useful for benchmarking
# the network in isolation). Has no effect for any other alignment_mode.
neural_refine_classically = true

# Automatically cull frames with low focus scores before stacking.
# GUI in-RAM path only (tile_size == 0, the default): the out-of-core tiled
# pipeline (used by the CLI always, and by the GUI when tile_size > 0) never
# holds the full aligned stack in memory (by design) and currently ignores
# this setting, logging a startup warning instead of silently no-op'ing. See
# the `auto_cull` field doc comment in `stacker_core::settings` for details.
auto_cull = true

# Sort frames by sharpness (centroid of in-focus detail) before stacking.
# This puts the sharpest frame first, making it the alignment reference.
# GUI in-RAM path only, identical limitations to auto_cull.
sort_by_sharpness = true

# Auto-Cull win-rate threshold: the minimum percentage (0.1..=5.0) of the
# scene's in-focus "detail" pixels a frame must win outright to survive
# culling. Frames that never uniquely win at least this share of detail
# pixels are treated as redundant near-duplicates and dropped. Only takes
# effect when auto_cull = true (GUI in-RAM path only).
auto_cull_threshold_pct = 2.0

# Crop the stacked output to the largest rectangle covered by every aligned
# frame, before fusion runs. Focus breathing means a warped frame's
# edge-clamped border replicates outer pixels into the band the warp doesn't
# actually cover; cropping it away removes that smeared band from the saved
# output AND excludes those pixels from fusion processing (culling scores,
# Apex pyramids, Relief SML/threshold statistics, AI). A rogue/misaligned frame
# that would shrink the crop below 25% of the canvas is ignored (falls back
# to the full canvas, with a warning logged).
# false always saves the full-canvas output (edge-clamped warp data
# visible at the borders).
crop_to_common_area = true

# When crop-to-common-area removed a border band, resample the cropped
# result back up to the original canvas resolution (high-quality 4-tap
# spline / Lanczos3, edge-clamped — no ringing or zero-fill at the border).
# The crop rectangle's aspect ratio can differ fractionally from the
# canvas, so this applies a slight non-uniform stretch — usually sub-pixel
# for typical focus-breathing crops. Ignored when no crop was applied, or
# when crop_to_common_area = false.
resize_cropped_to_original = false

# Out-of-core tiled processing (GUI only — the CLI always tiles via its own
# required --tile-size flag and does not read this setting).
# 0 (default) uses the GUI's normal in-RAM path: supports auto_cull and full
# live per-frame preview during fusion.
# > 0 switches the GUI to the same out-of-core tiled pipeline the CLI uses
# (tile edge length in pixels), for low-memory machines or very large/many-
# frame stacks. auto_cull and full live preview are unavailable in this mode.
tile_size = 0

# Process only every N-th frame from the loaded set (>= 1; 1 = all frames).
stack_every_nth = 1

# Allow the GPU-accelerated compute paths (Apex fusion, the production warp,
# Strata saliency, and the Relief guided-filter/multigrid engines) to engage.
# Only has any effect in a build compiled with the `gpu` Cargo feature; a
# default build has no wgpu dependency at all and always runs the CPU/rayon
# path regardless of this setting. Even when true (and an adapter is
# available), each accelerated stage still falls back to CPU transparently
# on any GPU failure — this only controls whether the GPU path is attempted.
use_gpu = true

# ── Apex (Laplacian-pyramid) ──────────────────────────────────────────────
# Number of pyramid levels used. Range: 2..=12.
pyramid_levels = 8

# Use all three color channels in the Apex selection metric instead of luma-only.
use_all_color_channels = false

# Use neighborhood-energy selection to suppress single-pixel "grit"/noise artifacts.
grit_suppression = true

# ── Relief (Depth-Map / SML) ──────────────────────────────────────────────
# Radius (in pixels) for the Sum-Modified-Laplacian focus measure. Range: 0..=100.
relief_estimation_radius = 5

# Post-processing smoothing radius for the depth map/index field. Range: 0..=50.
# Guided-filter engine: edge-preserving Guided Filter window radius.
# Multigrid engine: Gaussian-pyramid smoothing octave count.
relief_smoothing_radius = 2

# Percentile threshold for the depth-map focus-measure mask (0.0 to 1.0).
relief_contrast_pct = 0.0

# Show a live preview of the contrast mask during Relief stacking.
relief_show_preview = false

# Automatically compute the optimal contrast threshold when previewing.
relief_auto_detect = false

# Relief fusion engine.
#   false - guided-filter argmax (per-pixel focus-measure argmax + single
#           guided-filter smoothing pass on the final color)
#   true  - multigrid (confidence-weighted geometric multigrid V-cycle
#           depth-field solver)
# Both remain available; pick whichever suits your stack.
relief_use_multigrid = false

# ── Strata (Guided-Filter Fusion) ──────────────────────────────────────────
# Box-filter radius (pixels) separating the low-frequency "base" layer from
# the high-frequency "detail" layer before edge-aware guided-filter weight
# refinement. Range: 8..=64. The guided-filter radii/eps and blur sigma are
# fixed (see docs/strata-fusion-design.md for why).
strata_base_radius = 31

# "Detail focus" — the detail-layer weight re-concentration exponent applied
# after the per-frame guided-filter weight refinement, before accumulation.
# Range: 1..=5.
#   1 - original soft Li-Kang-Hu behaviour: best for flat/glossy subjects,
#       smoother transitions, fewer artifacts.
#   3 - (default) the deep-stack "watercolour" fix: keeps most depth-edge
#       crispness while staying tolerant of ambiguous transitions.
#   5 - near winner-take-all: crispest depth edges and the most detail
#       retention on deep (dozens-of-frames) stacks, at the cost of a
#       harder transition at ambiguous boundaries.
# Higher = crisper depth edges / more detail retention on deep stacks;
# lower = smoother, fewer artifacts on flat subjects.
strata_detail_focus = 3

# ── Preprocessing ─────────────────────────────────────────────────────────
[preprocessing]
# Clockwise rotation applied to every frame before stacking.
# Valid values: 0, 90, 180, 270.
pre_rotation = 0

# Enable pre-stacking center-crop.
pre_crop_enabled = false

# Crop specification string ("w,h" or "x,y,w,h").
pre_crop_spec = ""

# Resize all frames to this percentage of their original size before stacking.
# Range: 10..=100.
pre_resize_percent = 100

# Process frames in reverse file order before stacking.
sort_reverse = false

# Ignore EXIF orientation tags when loading frames.
ignore_exif_orientation = false

# ── Image Saving ──────────────────────────────────────────────────────────
[image_saving]
# Output image file format.
# Valid values: "tiff", "png", "jpeg"
output_format = "tiff"

# Bit depth used when writing the output file.
# Valid values: 8, 16.
bit_depth = 16

# Quality level for JPEG output. Range: 1..=100.
jpeg_quality = 95

# Template string for the output filename (without extension).
# '{name}' is replaced with the base name of the first input frame.
filename_template = "{name}_stacked"

# Default directory for saving results.
default_output_dir = ""

# Copy EXIF / XMP metadata from the first input frame to the output file.
copy_metadata = false
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_extension_matches_gui_save_dialog_mapping() {
        // Mirrors the `default_ext` mapping inlined in
        // `apps/stacker-gui/src/main.rs`'s `perform_save_current_image` —
        // this is the single source of truth both sides should agree with.
        assert_eq!(OutputFormat::Tiff.extension(), "tiff");
        assert_eq!(OutputFormat::Png.extension(), "png");
        assert_eq!(OutputFormat::Jpeg.extension(), "jpg");
    }
}
