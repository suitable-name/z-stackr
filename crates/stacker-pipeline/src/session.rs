//! The in-memory stack sequence: load, align, cull, reorder, fuse, crop.
//!
//! The desktop app and the browser worker both run a stack with every frame held
//! in memory (the CLI has its own out-of-core tile loop). The stages of that run
//! live here as plain functions: no Slint, no threads, no clock, no file system
//! (except [`PathSource`], which reads files). The callers keep the plumbing that
//! is theirs: the desktop its job thread and previews, the worker its messages.
//!
//! | Stage | Function |
//! |---|---|
//! | decode every frame | [`load_frames`] over a [`FrameSource`] ([`BytesSource`], [`PathSource`]) |
//! | align to the first frame | [`align_frames`] (or [`align_frames_with`] for a neural model) → [`AlignOutcome`] (crop, fallbacks, warning text) |
//! | auto-cull and sort | [`cull`] → [`CullOutcome`]; the standalone actions [`sort_frames_by_sharpness`], [`cull_frames`], [`run_sort_cull`] |
//! | fusion order | [`reorder`] |
//! | `Apex` | [`fuse_apex`], or [`fuse_apex_with_preview`] (progress and live previews from the streaming fold) |
//! | `Relief` | [`fuse_relief`] in one call, or [`relief_focus_max`] then [`relief_finish`] when a preview sits in between (guided filter, or the multigrid depth map with `relief_use_multigrid`, solved on the `Compute`) |
//! | `Strata` | [`fuse_strata`] (CPU), [`fuse_strata_with`] (the caller's kernels) or [`fuse_strata_compute`] (the `Compute`'s kernels) |
//! | `Relief` threshold | [`auto_contrast_pct`], [`threshold_for_pct`], [`threshold_from_settings`] |
//! | common valid area | [`crop_result`], or [`crop_result_for`] with `resize_cropped_to_original` |
//! | memory a run needs | [`estimate_peak_bytes`], [`fit_pre_resize`] |
//!
//! Every stage is the code the desktop app ran before this module existed, so
//! its results are unchanged bit for bit.
//!
//! # Which decoder the desktop calls
//!
//! [`load_frames`] decodes one frame after the other (the browser worker is
//! single-threaded). The desktop keeps its own `frames::decode_frames`, which
//! decodes in parallel with rayon and applies the same skip and size rules, and
//! uses this module from the alignment on.

use std::{convert::Infallible, fmt::Write as _, path::PathBuf};

use stacker_algo::{
    apex::fuse::{ApexFold, apex_fold_bytes_per_pixel},
    relief::threshold::{auto_detect_threshold, mask_above, percentile_threshold},
    strata::{CpuStrata, NO_PREVIEW, StrataKernels, strata_bytes_per_pixel},
};
use stacker_core::{
    image::PlanarImage,
    memory::extract_tile,
    settings::{AlignmentModeSetting, PreprocessingSettings, StackingSettings},
};

use crate::{
    Cancel, NoProgress, PipelineError, Progress, Stage,
    align::{
        AlignedFrame, AlignmentSettings, AsyncAlignBackend, AsyncSequenceError, Fallback,
        FrameFlow, SequenceError, align_sequence_async, align_sequence_on, lookahead_extra_bytes,
    },
    compute::Compute,
    crop::{Rect, resize_planar},
    cull::{CullRule, plan_frames_with},
    fraction_of,
    fuse::{self, BoxedStrata},
    load::{decode_frame, decode_frame_bytes, preprocessed_dimensions},
};

/// The most file names a message lists before it says "and N more".
const MAX_LISTED: usize = 5;

/// The file-name part of `name` (the text after the last `/` or `\`).
fn label_of(name: &str) -> &str {
    name.rsplit(['/', '\\']).next().unwrap_or(name)
}

/// `a.jpg, b.jpg, c.jpg and 4 more`: the file names of `names` (a path is
/// reduced to its last component), at most five of them.
#[must_use]
pub fn list_names<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let labels: Vec<&str> = names.into_iter().map(label_of).collect();
    let mut text = labels
        .iter()
        .take(MAX_LISTED)
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    if labels.len() > MAX_LISTED {
        let _ = write!(text, " and {} more", labels.len() - MAX_LISTED);
    }
    text
}

// ── Frame sources ───────────────────────────────────────────────────────────

/// Where the frames of a run come from.
pub trait FrameSource {
    /// Number of frames.
    fn len(&self) -> usize;

    /// `true` when there is no frame.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The name of frame `index` (a file name, or a path); used in messages and
    /// stays parallel to the decoded frames.
    fn name(&self, index: usize) -> &str;

    /// Decode frame `index` into the working representation.
    ///
    /// # Errors
    /// [`PipelineError::Decode`] when the frame cannot be decoded.
    fn decode(
        &mut self,
        index: usize,
        pre: &PreprocessingSettings,
    ) -> Result<PlanarImage<f32>, PipelineError>;
}

/// Frames held as encoded file contents (the browser: the page hands the worker
/// the bytes of each file it was given).
#[derive(Debug, Default, Clone)]
pub struct BytesSource {
    items: Vec<(String, Vec<u8>)>,
}

impl BytesSource {
    /// A source over `(name, file bytes)` pairs, in frame order.
    #[must_use]
    pub const fn new(items: Vec<(String, Vec<u8>)>) -> Self {
        Self { items }
    }
}

impl FrameSource for BytesSource {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn name(&self, index: usize) -> &str {
        &self.items[index].0
    }

    fn decode(
        &mut self,
        index: usize,
        pre: &PreprocessingSettings,
    ) -> Result<PlanarImage<f32>, PipelineError> {
        let (name, bytes) = &self.items[index];
        decode_frame_bytes(bytes, name, pre)
    }
}

/// Frames read from files.
#[derive(Debug, Default, Clone)]
pub struct PathSource {
    paths: Vec<PathBuf>,
    names: Vec<String>,
}

impl PathSource {
    /// A source over `paths`, in frame order. The name of a frame is its file
    /// name.
    #[must_use]
    pub fn new(paths: Vec<PathBuf>) -> Self {
        let names = paths
            .iter()
            .map(|p| {
                p.file_name().map_or_else(
                    || p.display().to_string(),
                    |n| n.to_string_lossy().into_owned(),
                )
            })
            .collect();
        Self { paths, names }
    }
}

impl FrameSource for PathSource {
    fn len(&self) -> usize {
        self.paths.len()
    }

    fn name(&self, index: usize) -> &str {
        &self.names[index]
    }

    fn decode(
        &mut self,
        index: usize,
        pre: &PreprocessingSettings,
    ) -> Result<PlanarImage<f32>, PipelineError> {
        decode_frame(&self.paths[index], pre)
    }
}

/// The decoded frames of one run. `names[i]` belongs to `frames[i]`;
/// `frames[0]` is the alignment reference.
#[derive(Debug, Default)]
pub struct LoadedFrames {
    /// The names of the frames that decoded, parallel to `frames`.
    pub names: Vec<String>,
    /// The decoded frames.
    pub frames: Vec<PlanarImage<f32>>,
    /// The frames that could not be decoded and were left out: name and the
    /// decoder's message.
    pub skipped: Vec<(String, String)>,
}

impl LoadedFrames {
    /// A short status suffix naming the frames that were left out, if any.
    #[must_use]
    pub fn skipped_note(&self) -> Option<String> {
        let (first, _) = self.skipped.first()?;
        let more = self.skipped.len() - 1;
        let tail = if more == 0 {
            String::new()
        } else {
            format!(" and {more} more")
        };
        Some(format!(
            "{} unreadable frame(s) skipped: {}{tail}",
            self.skipped.len(),
            label_of(first)
        ))
    }

    /// The one warning that names every frame that was left out, with the first
    /// decoder message as the likely common cause. `None` when nothing was
    /// skipped.
    #[must_use]
    pub fn skipped_warning(&self) -> Option<String> {
        let (_, first_reason) = self.skipped.first()?;
        Some(format!(
            "{} image(s) could not be read and were skipped: {} (first error: {first_reason})",
            self.skipped.len(),
            list_names(self.skipped.iter().map(|(name, _)| name.as_str())),
        ))
    }
}

/// The decoder's message without the "cannot decode frame '<path>'" prefix: the
/// skipped-frame note names the file itself.
fn decode_reason(error: &PipelineError) -> String {
    match error {
        PipelineError::Decode { message, .. } => message.clone(),
        other => other.to_string(),
    }
}

/// Decode every frame of `source`, in order.
///
/// A frame that fails to decode is left out (name and message in
/// [`LoadedFrames::skipped`]) and the run goes on; the first decoded frame is the
/// reference, and a later frame of a different size ends the run with an error
/// naming that file. Reports [`Stage::Load`] after every frame and polls
/// `cancel` before each.
///
/// # Errors
/// [`PipelineError::Cancelled`] when `cancel` fires, [`PipelineError::Invalid`]
/// when no frame decoded (or the source is empty) or a frame differs in size from
/// the reference.
pub fn load_frames<S: FrameSource + ?Sized>(
    source: &mut S,
    pre: &PreprocessingSettings,
    cancel: &dyn Cancel,
    progress: &dyn Progress,
) -> Result<LoadedFrames, PipelineError> {
    let attempted = source.len();
    let mut loaded = LoadedFrames::default();
    let mut first_reason: Option<String> = None;
    for index in 0..attempted {
        if cancel.is_cancelled() {
            return Err(PipelineError::Cancelled);
        }
        let name = source.name(index).to_owned();
        match source.decode(index, pre) {
            Ok(image) => {
                if let Some(reference) = loaded.frames.first() {
                    let expected = (reference.width, reference.height);
                    let found = (image.width, image.height);
                    if expected != found {
                        return Err(PipelineError::invalid(format!(
                            "{} is {}x{} px but the reference frame {} is {}x{} px; all frames \
                             must have the same size (check the pre-processing settings).",
                            label_of(&name),
                            found.0,
                            found.1,
                            label_of(&loaded.names[0]),
                            expected.0,
                            expected.1,
                        )));
                    }
                }
                loaded.names.push(name);
                loaded.frames.push(image);
            }
            Err(error) => {
                let reason = decode_reason(&error);
                tracing::warn!(name = name.as_str(), "failed to load frame: {reason}");
                first_reason.get_or_insert_with(|| reason.clone());
                loaded.skipped.push((name, reason));
            }
        }
        progress.stage(Stage::Load, fraction_of(index + 1, attempted), "");
    }
    if loaded.frames.is_empty() {
        return Err(PipelineError::invalid(if attempted == 0 {
            "No images to process: add some images first.".to_owned()
        } else {
            let reason = first_reason.map_or_else(String::new, |r| format!(" ({r})"));
            format!("None of the {attempted} image(s) could be loaded{reason}.")
        }));
    }
    Ok(loaded)
}

// ── Alignment ───────────────────────────────────────────────────────────────

/// What the alignment of one frame fell back to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackKind {
    /// The refinement failed; the frame was aligned from the starting estimate.
    Seed,
    /// The frame could not be warped; it was kept as loaded.
    Unaligned,
}

impl From<Fallback> for FallbackKind {
    fn from(fallback: Fallback) -> Self {
        match fallback {
            Fallback::Seed => Self::Seed,
            Fallback::Unaligned => Self::Unaligned,
        }
    }
}

/// A frame whose alignment did not go the normal way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameFallback {
    /// Position in the run's frame list.
    pub index: usize,
    /// What the frame fell back to.
    pub kind: FallbackKind,
}

/// The result of [`align_frames`].
#[derive(Debug, Default)]
pub struct AlignOutcome {
    /// The rectangle valid in every frame, `None` when no crop is needed.
    pub crop: Option<Rect>,
    /// The frames that fell back, in order.
    pub fallbacks: Vec<FrameFallback>,
    /// Why the GPU was not used for the alignment, or why it was given up
    /// part-way (`compute_device = "gpu"` only).
    pub device_notice: Option<String>,
}

impl AlignOutcome {
    /// One warning naming the frames (`names[i]` is frame `i`) that were not
    /// aligned properly, or `None` when every frame was.
    #[must_use]
    pub fn warning(&self, names: &[String]) -> Option<String> {
        let describe = |kind: FallbackKind| {
            let files: Vec<&str> = self
                .fallbacks
                .iter()
                .filter(|fallback| fallback.kind == kind)
                .filter_map(|fallback| names.get(fallback.index))
                .map(String::as_str)
                .collect();
            (!files.is_empty()).then(|| list_names(files))
        };
        let mut parts = Vec::new();
        if let Some(files) = describe(FallbackKind::Seed) {
            parts.push(format!(
                "Alignment could not be refined for {files}; the starting estimate was used, so \
                 they may be misaligned."
            ));
        }
        if let Some(files) = describe(FallbackKind::Unaligned) {
            parts.push(format!(
                "Could not warp {files}; kept as loaded, unaligned."
            ));
        }
        if let Some(notice) = &self.device_notice {
            parts.push(notice.clone());
        }
        (!parts.is_empty()).then(|| parts.join(" "))
    }
}

/// The in-memory end of the shared alignment driver: frames move out of the list
/// to be aligned and back in, and every aligned frame is announced to `on_frame`
/// together with the running common-coverage mask.
struct MemoryFlow<'a> {
    frames: &'a mut [PlanarImage<f32>],
    on_frame: &'a mut OnFrame<'a>,
}

/// The per-frame callback of [`align_frames`]: frame index, aligned frame, mask.
type OnFrame<'a> = dyn FnMut(usize, &PlanarImage<f32>, &[bool]) + 'a;

impl FrameFlow<Infallible> for MemoryFlow<'_> {
    fn load(&mut self, index: usize) -> Result<PlanarImage<f32>, Infallible> {
        // The frame moves out (no copy); `deliver` puts its aligned version back.
        Ok(std::mem::replace(
            &mut self.frames[index],
            PlanarImage::new(0, 0),
        ))
    }

    fn deliver(
        &mut self,
        index: usize,
        aligned: AlignedFrame,
        common: &[bool],
    ) -> Result<(), Infallible> {
        self.frames[index] = aligned.frame;
        (self.on_frame)(index, &self.frames[index], common);
        Ok(())
    }

    fn reload(&mut self, index: usize) -> Option<Result<PlanarImage<f32>, Infallible>> {
        // The neural pre-pass reads every frame once before the first is aligned
        // (and so moved out): hand it a copy, the frame stays where it is.
        self.frames.get(index).map(|frame| Ok(frame.clone()))
    }
}

/// Align every frame to `frames[0]` in place with the shared aligner.
///
/// Returns the crop rectangle valid in every frame (none with
/// `crop_to_common_area` off: the border the alignment left stays in the
/// result). Does nothing (and returns an empty outcome) when alignment is off or
/// there is only one frame.
///
/// `on_frame(i, aligned, mask)` is called after frame `i` is aligned, with the
/// running common-coverage mask. The outcome lists the frames that fell back to
/// the seed or stayed unwarped, so the caller can tell the user.
///
/// The objective and the warp run on `compute` (the GPU when it has a backend,
/// the CPU otherwise). `cancel` is polled before every frame; when it fires the
/// result is `None` (the frames are then partly aligned and must be discarded).
///
/// The alignment settings are `AlignmentSettings::from(settings)`; a caller with
/// a neural alignment model uses [`align_frames_with`].
pub fn align_frames(
    frames: &mut [PlanarImage<f32>],
    settings: &StackingSettings,
    compute: &mut Compute,
    cancel: &dyn Cancel,
    on_frame: &mut OnFrame<'_>,
) -> Option<AlignOutcome> {
    align_frames_with(
        frames,
        settings,
        &AlignmentSettings::from(settings),
        compute,
        cancel,
        on_frame,
    )
}

/// [`align_frames`] with a neural alignment `model` (`None` is the plain call).
///
/// It is `AlignmentSettings::from(settings).with_model(model)`. The crop honours
/// `crop_to_common_area`; apply `resize_cropped_to_original` to the fused result
/// with [`crop_result_for`].
pub fn align_frames_with_model(
    frames: &mut [PlanarImage<f32>],
    settings: &StackingSettings,
    model: Option<crate::align::AlignModel>,
    compute: &mut Compute,
    cancel: &dyn Cancel,
    on_frame: &mut OnFrame<'_>,
) -> Option<AlignOutcome> {
    align_frames_with(
        frames,
        settings,
        &AlignmentSettings::from(settings).with_model(model),
        compute,
        cancel,
        on_frame,
    )
}

/// [`align_frames`] with the alignment settings of the caller.
///
/// For instance `AlignmentSettings::from(settings).with_model(Some(model))` for
/// `alignment_mode = "neural"`. `settings` still decides `alignment_mode ==
/// none` (nothing to do) and `crop_to_common_area`.
pub fn align_frames_with(
    frames: &mut [PlanarImage<f32>],
    settings: &StackingSettings,
    alignment: &AlignmentSettings,
    compute: &mut Compute,
    cancel: &dyn Cancel,
    on_frame: &mut OnFrame<'_>,
) -> Option<AlignOutcome> {
    if settings.alignment_mode == AlignmentModeSetting::None || frames.len() <= 1 {
        return Some(AlignOutcome::default());
    }
    let count = frames.len();
    // The reference moves out of the list while the others are aligned against
    // it (no 12 B/px clone) and goes back afterwards.
    let reference = std::mem::replace(&mut frames[0], PlanarImage::new(0, 0));
    let (width, height) = (reference.width, reference.height);

    let mut flow = MemoryFlow { frames, on_frame };
    let result = align_sequence_on(
        &reference,
        count,
        alignment,
        cancel,
        &NoProgress,
        &mut flow,
        compute,
    );
    flow.frames[0] = reference;

    match result {
        Ok(outcome) => Some(AlignOutcome {
            crop: outcome
                .coverage
                .crop_when(settings.crop_to_common_area, |i| format!("frame {i}"))
                .needed(width, height),
            fallbacks: outcome
                .fallbacks
                .into_iter()
                .map(|(index, kind)| FrameFallback {
                    index,
                    kind: kind.into(),
                })
                .collect(),
            device_notice: compute.take_notice(),
        }),
        Err(SequenceError::Cancelled) => None,
        Err(SequenceError::Flow(never)) => match never {},
    }
}

/// [`align_frames`] on an asynchronous device (the browser's WebGPU): the same
/// in-place alignment to `frames[0]`, with the refiners as resumable searches and
/// the warp awaited, and with no CPU to fall back on.
///
/// `Ok(None)` when `cancel` fired (the frames are then partly aligned and must be
/// discarded); an empty outcome when alignment is off or there is one frame.
///
/// # Errors
/// [`PipelineError::Invalid`] ("The GPU stopped during alignment …") when the
/// device declined a level, failed an evaluation or failed the warp.
pub async fn align_frames_async<B: AsyncAlignBackend>(
    frames: &mut [PlanarImage<f32>],
    settings: &StackingSettings,
    backend: &mut B,
    cancel: &dyn Cancel,
    on_frame: &mut OnFrame<'_>,
) -> Result<Option<AlignOutcome>, PipelineError> {
    if settings.alignment_mode == AlignmentModeSetting::None || frames.len() <= 1 {
        return Ok(Some(AlignOutcome::default()));
    }
    let count = frames.len();
    // The reference moves out of the list while the others are aligned against
    // it and goes back afterwards (as in `align_frames`).
    let reference = std::mem::replace(&mut frames[0], PlanarImage::new(0, 0));
    let (width, height) = (reference.width, reference.height);

    let mut flow = MemoryFlow { frames, on_frame };
    let result = align_sequence_async(
        &reference,
        count,
        &AlignmentSettings::from(settings),
        cancel,
        &NoProgress,
        &mut flow,
        backend,
        false,
    )
    .await;
    flow.frames[0] = reference;

    match result {
        Ok(outcome) => Ok(Some(AlignOutcome {
            crop: outcome
                .coverage
                .crop_when(settings.crop_to_common_area, |i| format!("frame {i}"))
                .needed(width, height),
            fallbacks: outcome
                .fallbacks
                .into_iter()
                .map(|(index, kind)| FrameFallback {
                    index,
                    kind: kind.into(),
                })
                .collect(),
            device_notice: None,
        })),
        Err(AsyncSequenceError::Cancelled) => Ok(None),
        Err(AsyncSequenceError::Flow(never)) => match never {},
        Err(AsyncSequenceError::Device(error)) => Err(PipelineError::invalid(format!(
            "The GPU stopped during alignment ({error}). In the browser, alignment runs on the \
             GPU only."
        ))),
    }
}

// ── Cull and order ──────────────────────────────────────────────────────────

/// The decision of the auto-cull for a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CullOutcome {
    /// The kept frames (positions in the frame list) in fusion order; in range,
    /// unique and never empty.
    pub order: Vec<usize>,
    /// The positions that are not in `order`, ascending.
    pub dropped: Vec<usize>,
    /// Notes for the user: no detail at all, or a cull that would have emptied
    /// the stack.
    pub warnings: Vec<String>,
}

impl CullOutcome {
    /// The warnings as one message, or `None` when there are none.
    #[must_use]
    pub fn warning(&self) -> Option<String> {
        (!self.warnings.is_empty()).then(|| format!("Auto-cull: {}", self.warnings.join("; ")))
    }
}

/// Decide which frames enter the fusion and in which order.
///
/// * `auto_cull` drops the frames that win too little of the in-focus detail.
///   The threshold is `auto_cull_threshold_pct` when that is above 0, else the
///   rule that scales with the frame count (`min(2, 50 / N)` percent). The CLI
///   applies the same census and threshold.
/// * `sort_by_sharpness` fuses the kept frames in the census (nearest-neighbour)
///   order; without it they keep their stack order.
/// * With neither, every frame is kept in list order.
///
/// # Errors
/// [`PipelineError::Invalid`] ("Auto-cull left no frames to stack.") when the
/// decision leaves no frame.
pub fn cull(
    frames: &[PlanarImage<f32>],
    settings: &StackingSettings,
) -> Result<CullOutcome, PipelineError> {
    run_sort_cull(
        frames,
        settings.sort_by_sharpness,
        settings.auto_cull,
        settings,
    )
}

/// The order of all frames by the census of the focus measure (the order
/// `sort_by_sharpness` fuses in), without dropping any: the GUI's Sort action.
///
/// The result is a permutation of the frame positions. With no detail at all the
/// order is the stack order.
///
/// # Errors
/// [`PipelineError::Invalid`] when there is no frame.
pub fn sort_frames_by_sharpness(
    frames: &[PlanarImage<f32>],
    settings: &StackingSettings,
) -> Result<CullOutcome, PipelineError> {
    run_sort_cull(frames, true, false, settings)
}

/// Drop the frames that win too little of the in-focus detail: the GUI's Cull
/// action.
///
/// The `auto_cull` and `sort_by_sharpness` toggles do not matter (the threshold
/// rule comes from `auto_cull_threshold_pct`); the kept frames stay in stack
/// order.
///
/// # Errors
/// [`PipelineError::Invalid`] when there is no frame or the decision leaves none.
pub fn cull_frames(
    frames: &[PlanarImage<f32>],
    settings: &StackingSettings,
) -> Result<CullOutcome, PipelineError> {
    run_sort_cull(frames, false, true, settings)
}

/// The one implementation behind [`cull`], [`sort_frames_by_sharpness`] and
/// [`cull_frames`]; the toggles are explicit so the GUI's Sort and Cull buttons
/// can combine them independently of the settings.
///
/// * neither: every frame in list order;
/// * `sort_by_sharpness` only: all frames, in census order (nothing is dropped);
/// * `auto_cull` only: the survivors in stack order;
/// * both: the survivors in census order.
///
/// # Errors
/// [`PipelineError::Invalid`] ("Auto-cull left no frames to stack.") when the
/// decision leaves no frame, or there is no frame at all.
pub fn run_sort_cull(
    frames: &[PlanarImage<f32>],
    sort_by_sharpness: bool,
    auto_cull: bool,
    settings: &StackingSettings,
) -> Result<CullOutcome, PipelineError> {
    if !sort_by_sharpness && !auto_cull {
        return Ok(CullOutcome {
            order: (0..frames.len()).collect(),
            dropped: Vec::new(),
            warnings: Vec::new(),
        });
    }
    let rule = if auto_cull {
        CullRule::from_settings(settings)
    } else {
        CullRule::Fixed(0.0)
    };
    let plan = plan_frames_with(frames, rule);

    // In range and unique: `reorder` moves each frame at most once.
    let mut seen = vec![false; frames.len()];
    let order: Vec<usize> = plan
        .fusion_order(sort_by_sharpness)
        .into_iter()
        .filter(|&i| i < frames.len() && !std::mem::replace(&mut seen[i], true))
        .collect();
    if order.is_empty() {
        return Err(PipelineError::invalid("Auto-cull left no frames to stack."));
    }
    // `seen` marks exactly the positions `order` holds.
    let dropped = seen
        .iter()
        .enumerate()
        .filter(|&(_, &kept)| !kept)
        .map(|(i, _)| i)
        .collect();
    Ok(CullOutcome {
        order,
        dropped,
        warnings: plan.warnings,
    })
}

/// `frames` in `order`. Frames move: nothing is cloned, and the frames `order`
/// leaves out (culled, or repeated indices) are freed on return, before the
/// fusion allocates its own buffers.
#[must_use]
pub fn reorder(frames: Vec<PlanarImage<f32>>, order: &[usize]) -> Vec<PlanarImage<f32>> {
    let mut slots: Vec<Option<PlanarImage<f32>>> = frames.into_iter().map(Some).collect();
    order
        .iter()
        .filter_map(|&idx| slots.get_mut(idx)?.take())
        .collect()
}

// ── Fusion ──────────────────────────────────────────────────────────────────

/// `Apex` fusion through `compute` (the GPU when it has a backend and the call
/// fits, the CPU otherwise). Never fails.
///
/// On the CPU the frames are folded into one accumulator pyramid, one at a time
/// (`stacker_algo::apex::fuse::ApexFold`): the result equals the batch fusion of
/// all the frames' pyramids bit for bit, and the memory beyond the frames is
/// `apex_fold_bytes_per_pixel(levels)` whatever the frame count.
#[must_use]
pub fn fuse_apex(
    compute: &mut Compute,
    frames: &[PlanarImage<f32>],
    settings: &StackingSettings,
) -> PlanarImage<f32> {
    compute.apex(frames, settings)
}

/// [`fuse_apex`] reporting `Stage::Fuse` progress (one tick per folded frame) and
/// live previews.
///
/// On the CPU `on_preview` gets the fusion of the frames folded so far
/// (`ApexFold::preview`: one reconstruction, no copy of the accumulator) after
/// every `preview_every` frames, never after the last one, and once more with the
/// finished image (`stacker_algo::strata::NO_PREVIEW` switches previews off). On a
/// GPU backend the fusion is one device call: only the finished image is
/// announced.
#[must_use]
pub fn fuse_apex_with_preview(
    compute: &mut Compute,
    frames: &[PlanarImage<f32>],
    settings: &StackingSettings,
    progress: &dyn Progress,
    preview_every: usize,
    mut on_preview: impl FnMut(&PlanarImage<f32>),
) -> PlanarImage<f32> {
    if frames.is_empty() {
        return PlanarImage::new(0, 0);
    }
    progress.stage(Stage::Fuse, 0.0, "Apex");
    let previews = preview_every != NO_PREVIEW;
    let every = preview_every.max(1);
    let total = frames.len();
    let fused = if compute.uses_gpu() {
        compute.apex(frames, settings)
    } else {
        let mut fold = ApexFold::new(
            settings.pyramid_levels as usize,
            settings.use_all_color_channels,
            settings.grit_suppression,
        );
        fold.fold_source(frames, |done, fold| {
            progress.stage(Stage::Fuse, fraction_of(done, total), "");
            if previews && done < total && done % every == 0 {
                on_preview(&fold.preview());
            }
        });
        fold.finish()
    };
    progress.stage(Stage::Fuse, 1.0, "");
    if previews {
        on_preview(&fused);
    }
    fused
}

/// `Strata` fusion on the CPU, reporting `Stage::Fuse` progress (`2 × frames`
/// ticks: the saliency pass, then the accumulation pass) and live previews.
///
/// `on_preview` gets a snapshot of the composite after every `preview_every`
/// frames of the second pass and once more with the finished image
/// (`stacker_algo::strata::NO_PREVIEW` switches previews off). A snapshot costs one
/// extra pass over the image. The stage cannot be cancelled part-way; the caller
/// polls its cancel flag between the passes of a run.
///
/// # Panics
/// Panics when there are more than 65 536 frames or the frames differ in size.
#[must_use]
pub fn fuse_strata(
    frames: &[PlanarImage<f32>],
    settings: &StackingSettings,
    progress: &dyn Progress,
    preview_every: usize,
    on_preview: impl FnMut(&PlanarImage<f32>),
) -> PlanarImage<f32> {
    fuse_strata_with(
        &mut CpuStrata,
        frames,
        settings,
        progress,
        preview_every,
        on_preview,
    )
}

/// [`fuse_strata`] with the caller's kernels: the seam for a device.
///
/// A `StrataKernels` implementation serves the saliency and the weight refinement;
/// what it declines runs on the CPU (see `stacker_algo::strata::StrataKernels`).
///
/// # Panics
/// As [`fuse_strata`].
#[must_use]
pub fn fuse_strata_with<K: StrataKernels>(
    kernels: &mut K,
    frames: &[PlanarImage<f32>],
    settings: &StackingSettings,
    progress: &dyn Progress,
    preview_every: usize,
    on_preview: impl FnMut(&PlanarImage<f32>),
) -> PlanarImage<f32> {
    if frames.is_empty() {
        return PlanarImage::new(0, 0);
    }
    let total = frames.len() * 2;
    let mut done = 0_usize;
    progress.stage(Stage::Fuse, 0.0, "Strata");
    fuse::strata_on(
        kernels,
        frames,
        settings,
        || {
            done += 1;
            progress.stage(Stage::Fuse, fraction_of(done, total), "");
        },
        preview_every,
        on_preview,
    )
}

/// [`fuse_strata`] on the kernels `compute` offers.
///
/// These are the device's saliency and guided-filter kernels when the handle has a backend, `CpuStrata` otherwise
/// ([`Compute::strata_kernels`]). Afterwards the handle is told to check the
/// kernels ([`Compute::check_strata_kernels`]): a device fault during the run is
/// given up there and shows in `compute.take_notice()`; the fusion itself has
/// carried on on the CPU and its result stands.
///
/// # Panics
/// As [`fuse_strata`].
#[must_use]
pub fn fuse_strata_compute(
    compute: &mut Compute,
    frames: &[PlanarImage<f32>],
    settings: &StackingSettings,
    progress: &dyn Progress,
    preview_every: usize,
    on_preview: impl FnMut(&PlanarImage<f32>),
) -> PlanarImage<f32> {
    let mut kernels = BoxedStrata(compute.strata_kernels());
    let fused = fuse_strata_with(
        &mut kernels,
        frames,
        settings,
        progress,
        preview_every,
        on_preview,
    );
    compute.check_strata_kernels();
    fused
}

/// `Relief` fusion of a whole image in one call.
///
/// `resolve` decides the absolute contrast threshold from the maximum focus map
/// (in its `luma` plane) and is called exactly once; `None` from it (the user
/// cancelled) is passed on.
///
/// With `|max| Some(threshold_from_settings(max, settings))` this is
/// `fuse::relief`. With `relief_use_multigrid` the focus maps are computed on the
/// CPU and the depth map is solved on `compute` (see [`relief_focus_max`]).
pub fn fuse_relief(
    compute: &mut Compute,
    frames: &[PlanarImage<f32>],
    settings: &StackingSettings,
    resolve: impl FnOnce(&PlanarImage<f32>) -> Option<f32>,
) -> Option<PlanarImage<f32>> {
    if settings.relief_use_multigrid
        && !frames.is_empty()
        && let Ok(focus) = relief_focus_max(compute, frames, settings)
    {
        let threshold = resolve(focus.max())?;
        return Some(relief_finish(compute, frames, settings, focus, threshold));
    }
    compute.relief(frames, settings, resolve)
}

/// The first half of a `Relief` fusion that stops for a decision.
///
/// It holds the maximum focus map and, on the CPU, the index of the winning frame
/// of every pixel, which [`relief_finish`] reuses: 6 bytes per pixel in all,
/// whatever the frame count (the per-frame focus maps are not kept). With a
/// device the maximum map alone is held (the device keeps the stack it folded).
/// With more than [`fuse::MAX_MULTIGRID_FRAMES`] frames on the CPU the per-frame
/// maps are kept instead (4 bytes per pixel and frame).
#[derive(Debug)]
pub struct ReliefFocus {
    max: PlanarImage<f32>,
    kind: FocusKind,
}

/// What a [`ReliefFocus`] holds besides the maximum map.
#[derive(Debug)]
enum FocusKind {
    /// Nothing: the device (or an empty stack).
    Device,
    /// The winner index of the streamed votes, for the guided selection.
    Votes(Vec<u16>),
    /// One focus map per frame (a stack too deep for a `u16` winner index).
    Maps(Vec<Vec<f32>>),
    /// The winner index for the multigrid depth map.
    Multigrid(Vec<u16>),
}

impl ReliefFocus {
    /// The best focus value per pixel, in the `luma` plane.
    #[must_use]
    pub const fn max(&self) -> &PlanarImage<f32> {
        &self.max
    }
}

/// Compute the maximum focus map of `frames` (the first half of [`fuse_relief`]).
///
/// The caller decides the threshold (see [`threshold_for_pct`] and
/// [`threshold_from_settings`]), then calls [`relief_finish`] with the same frames.
///
/// With `relief_use_multigrid` the focus maps are computed on the CPU one frame
/// at a time (the GPU has no kernel for the winner index) and only the maximum
/// map and the winning frame index are kept. The frames must be in capture order
/// and number at most [`fuse::MAX_MULTIGRID_FRAMES`].
///
/// # Errors
/// [`PipelineError::Core`] if the GPU returned a focus map of the wrong size,
/// [`PipelineError::Invalid`] for a multigrid run over more than 65 536 frames.
pub fn relief_focus_max(
    compute: &mut Compute,
    frames: &[PlanarImage<f32>],
    settings: &StackingSettings,
) -> Result<ReliefFocus, PipelineError> {
    let Some(first) = frames.first() else {
        return Ok(ReliefFocus {
            max: PlanarImage::new(0, 0),
            kind: FocusKind::Device,
        });
    };
    let radius = settings.relief_estimation_radius as usize;
    if settings.relief_use_multigrid {
        if frames.len() > fuse::MAX_MULTIGRID_FRAMES {
            return Err(PipelineError::invalid(format!(
                "The multigrid depth map handles at most {} frames, but the stack has {}.",
                fuse::MAX_MULTIGRID_FRAMES,
                frames.len()
            )));
        }
        let (best, index) = fuse::multigrid_votes(frames, radius).into_parts();
        return Ok(ReliefFocus {
            max: PlanarImage::with_luma(first.width, first.height, best)?,
            kind: FocusKind::Multigrid(index),
        });
    }
    if compute.uses_gpu() {
        let max = compute.relief_focus_max(frames, radius);
        return Ok(ReliefFocus {
            max: PlanarImage::with_luma(first.width, first.height, max)?,
            kind: FocusKind::Device,
        });
    }
    if frames.len() > fuse::MAX_MULTIGRID_FRAMES {
        let maps = fuse::focus_maps(frames, radius);
        return Ok(ReliefFocus {
            max: maps.max,
            kind: FocusKind::Maps(maps.smls),
        });
    }
    // The CPU folds each frame's focus map into the best value and the winner as
    // it is computed: 6 B/px instead of 4 B/px per frame.
    let (best, index) = fuse::focus_votes(frames, radius).into_parts();
    Ok(ReliefFocus {
        max: PlanarImage::with_luma(first.width, first.height, best)?,
        kind: FocusKind::Votes(index),
    })
}

/// Select and smooth the stack against the absolute contrast `threshold`.
///
/// This is the second half of [`fuse_relief`]. `frames` must be the frames [`relief_focus_max`]
/// was computed from. For the same threshold the result equals [`fuse_relief`]'s,
/// bit for bit. With `relief_use_multigrid` the depth map is solved here, on
/// `compute` (the GPU solver when the handle has a backend that serves it, the CPU
/// solver otherwise; about 28 bytes per pixel on the CPU on top of the frames, 4
/// more for the depth) and the frames are interpolated along it.
#[must_use]
pub fn relief_finish(
    compute: &mut Compute,
    frames: &[PlanarImage<f32>],
    settings: &StackingSettings,
    focus: ReliefFocus,
    threshold: f32,
) -> PlanarImage<f32> {
    if frames.is_empty() {
        return PlanarImage::new(0, 0);
    }
    let smoothing = settings.relief_smoothing_radius as usize;
    match focus.kind {
        FocusKind::Multigrid(index) => {
            let depth = fuse::multigrid_depth_map_on(
                compute,
                (focus.max.width, focus.max.height),
                &focus.max.luma,
                &index,
                threshold,
                settings,
            );
            fuse::relief_multigrid_tile(frames, &depth)
        }
        FocusKind::Votes(index) => {
            fuse::select_from_votes(frames, &focus.max.luma, &index, threshold, smoothing)
        }
        FocusKind::Maps(smls) => {
            let mask = mask_above(&focus.max.luma, threshold);
            fuse::select_and_smooth(frames, &smls, &mask, smoothing)
        }
        FocusKind::Device => compute.relief_tile(frames, settings, threshold),
    }
}

/// The auto-detected rejected fraction (`0.0..=1.0`) of a maximum focus map.
///
/// It follows the median + MAD noise-floor rule of `auto_detect_threshold` and
/// is the value the preview starts at with `relief_auto_detect`. It is the number
/// `compute_relief_threshold` turns into the threshold, so the preview and the
/// run agree.
#[must_use]
pub fn auto_contrast_pct(focus_max: &PlanarImage<f32>) -> f32 {
    auto_detect_threshold(&focus_max.luma)
}

/// The absolute threshold for rejecting the fraction `contrast_pct` of the
/// pixels of a maximum focus map (the preview slider's value).
#[must_use]
pub fn threshold_for_pct(focus_max: &PlanarImage<f32>, contrast_pct: f32) -> f32 {
    percentile_threshold(&focus_max.luma, contrast_pct)
}

/// The absolute threshold the settings ask for on a maximum focus map:
/// `relief_contrast_pct`, or the auto-detected noise floor with
/// `relief_auto_detect`.
#[must_use]
pub fn threshold_from_settings(focus_max: &PlanarImage<f32>, settings: &StackingSettings) -> f32 {
    fuse::relief_threshold(&focus_max.luma, settings).value
}

/// How many thresholds a `Relief` preview carries: the rejected fractions
/// `0/1000, 1/1000 ... 1000/1000`.
pub const THRESHOLD_TABLE_LEN: usize = 1001;

/// The absolute thresholds of a focus map for the rejected fractions
/// `k / 1000`, `k = 0..=1000` ([`THRESHOLD_TABLE_LEN`] values).
///
/// Entry `k` is exactly what [`percentile_threshold`] returns for the fraction
/// `k / 1000` (the rank is rounded, ties between "at least" and "above" resolve to
/// the closer count), so a preview slider that snaps to the table fuses with the
/// same mask the CLI would. One sort serves all entries. An empty map gives all
/// zeros.
#[must_use]
pub fn threshold_table(values: &[f32]) -> Vec<f32> {
    let len = values.len();
    if len == 0 {
        return vec![0.0; THRESHOLD_TABLE_LEN];
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    (0..=1000_u16)
        .map(|step| {
            // The fraction is an `f32`, as the preview slider and the shared rule
            // see it (0.9 is really 0.89999998...), so the rank rounds the same.
            let pct = f64::from(f32::from(step) / 1000.0);
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the rank is a rounded fraction of a pixel count, clamped to the map"
            )]
            let k = ((len as f64 * pct).round() as usize).min(len - 1);
            let t = sorted[k];
            let wanted = len - k;
            let at_least = len - sorted.partition_point(|&v| v.total_cmp(&t).is_lt());
            let above = len - sorted.partition_point(|&v| v.total_cmp(&t).is_le());
            if above.abs_diff(wanted) < at_least.abs_diff(wanted) {
                t.next_up()
            } else {
                t
            }
        })
        .collect()
}

/// Cut a fused image to the common valid area of the aligned frames (`None`: the
/// image as it is), removing the warp borders.
#[must_use]
pub fn crop_result(fused: PlanarImage<f32>, crop: Option<Rect>) -> PlanarImage<f32> {
    match crop {
        Some((cx, cy, cw, ch)) => extract_tile(&fused, cx, cy, cw, ch),
        None => fused,
    }
}

/// [`crop_result`] honouring `resize_cropped_to_original`.
///
/// After the crop the
/// image is stretched back to the size of `fused` (the canvas the frames were
/// aligned on) with the edge-clamped Catmull-Rom resampler of
/// [`crate::crop::resize_planar`]. Nothing is resized when no crop was needed
/// (`crop` is `None`) or the setting is off, so with the setting off this is
/// exactly [`crop_result`].
#[must_use]
pub fn crop_result_for(
    fused: PlanarImage<f32>,
    crop: Option<Rect>,
    settings: &StackingSettings,
) -> PlanarImage<f32> {
    let (width, height) = (fused.width, fused.height);
    let cropped = crop_result(fused, crop);
    if crop.is_some() && settings.resize_cropped_to_original {
        resize_planar(cropped, width, height)
    } else {
        cropped
    }
}

// ── Memory ──────────────────────────────────────────────────────────────────

/// A fusion pass of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PassKind {
    /// Laplacian-pyramid maximum contrast.
    Apex,
    /// Depth map (guided filter, or the multigrid solver with
    /// `relief_use_multigrid`).
    Relief,
    /// Guided-filter soft blend.
    Strata,
}

/// The smallest Pre-resize percentage the settings allow
/// (`preprocessing.pre_resize_percent` is clamped to `10..=100`).
pub const MIN_PRE_RESIZE_PERCENT: u32 = 10;

/// The Pre-resize percentages [`fit_pre_resize`] steps by.
const PRE_RESIZE_STEP: u32 = 5;

/// Bytes of one planar `f32` frame per pixel (`luma`, `chroma_a`, `chroma_b`).
const FRAME_BYTES: u64 = 12;
/// Transient bytes per source pixel while one frame decodes (the widest decoded
/// image, RGBA 16-bit).
const DECODE_SOURCE_BYTES: u64 = 8;
/// Transient bytes per working pixel while one frame decodes (the resized copy).
const DECODE_WORKING_BYTES: u64 = 8;
/// Extra bytes per pixel while aligning, beyond the frames: the aligned frame
/// (12), the refinement working set (two luma pyramids and a warped luma plane,
/// about 15) and the coverage masks (2).
const ALIGN_WORKING_BYTES: u64 = 29;
/// Bytes per pixel and frame while culling: the frame (12) and its focus map (4).
const CULL_FRAME_BYTES: u64 = 16;
/// Extra bytes per pixel while culling: the winner map (an `f32` and a `u32`).
const CULL_WORKING_BYTES: u64 = 8;
/// Extra bytes per pixel while aligning with `chain_alignment`: the raw previous
/// frame (12) and its luma pyramid (4 · 4/3, rounded up to 6).
pub(crate) const CHAIN_WORKING_BYTES: u64 = FRAME_BYTES + 6;
/// Bytes per pixel and frame during `Apex`: the frame only. The frames are folded
/// into one accumulator pyramid one at a time, so no pyramid is kept per frame.
const APEX_FRAME_BYTES: u64 = FRAME_BYTES;
/// Bytes per pixel and frame during `Relief`: the frame only. The focus maps are
/// folded into the best value and the winner as they are computed, so no map is
/// kept per frame.
const RELIEF_FRAME_BYTES: u64 = FRAME_BYTES;
/// Extra bytes per pixel during `Relief`: the maximum map (4), the winning frame
/// (2), the mask, the selected image, the guided filter's temporaries and the
/// result (the 60 of the per-frame-map path, which this one undercuts by the
/// focus maps it no longer holds).
const RELIEF_WORKING_BYTES: u64 = 62;
/// Extra bytes per pixel during `Relief` with the multigrid solver, on top of the
/// guided figures above (which stay an upper bound for the focus stage): the
/// solver's peak (28), the depth map (4) and the best-focus map it still holds
/// for the threshold (4).
const MULTIGRID_EXTRA_BYTES: u64 = 36;
/// Bytes per pixel and frame during `Strata`: the frame. Its working set does not
/// grow with the frame count.
const STRATA_FRAME_BYTES: u64 = FRAME_BYTES;
/// Extra bytes per pixel during `Strata` (`strata_bytes_per_pixel()`, 150): the
/// accumulators, the winner map, two frames' prepared planes, the refined weights,
/// the guided-filter scratch and the output or preview snapshot.
const STRATA_WORKING_BYTES: u64 = strata_bytes_per_pixel() as u64;

/// `true` when the alignment of a run has a seed source (AKAZE matches, or the
/// network's matrices): only then does the look-ahead hold more than one frame.
pub(crate) fn has_seed_source(settings: &StackingSettings) -> bool {
    settings.alignment_mode == AlignmentModeSetting::Neural
        || (cfg!(feature = "akaze") && settings.akaze_seeding)
}

/// Bytes the alignment look-ahead holds beyond the frame in flight for a
/// `width × height` working size (see `align::lookahead_extra_bytes`): the
/// window's frames with a seed source (none on `wasm32`), plus the frame the
/// second thread of the overlapped driver decodes (`overlapped`). The in-memory
/// driver of this module is not the overlapped one.
pub(crate) fn lookahead_bytes(
    width: usize,
    height: usize,
    settings: &StackingSettings,
    overlapped: bool,
) -> u64 {
    if has_seed_source(settings) {
        return u64::try_from(lookahead_extra_bytes(width, height, overlapped)).unwrap_or(u64::MAX);
    }
    if overlapped {
        // No seed source: the window is one frame wide, the decode thread's
        // frame is all that comes on top.
        let pixels = u64::try_from(width)
            .unwrap_or(u64::MAX)
            .saturating_mul(u64::try_from(height).unwrap_or(u64::MAX));
        return pixels.saturating_mul(FRAME_BYTES);
    }
    0
}

/// The peak memory, in bytes, of an in-memory run.
///
/// The run is over `frames` frames of `width × height` source pixels; the peak is
/// the largest of the load, align, cull and fusion stages (see the crate README,
/// "Desktop app (in-memory)"). The align stage adds the look-ahead's frames when
/// the alignment has a seed source (`lookahead_extra_bytes`) and the previous
/// frame with its pyramid for `chain_alignment`. `Apex` costs
/// `apex_fold_bytes_per_pixel(pyramid_levels)` (84 at 8 levels) beyond the frames,
/// whatever their number; `Relief` 12 per frame plus 62.
///
/// The size after `settings.preprocessing` (rotation, crop, resize) is the size
/// every stage works at. `frames` counts the frames that enter the run (after
/// `stack_every_nth`); the cull is assumed to keep all of them, and the results of
/// earlier passes are assumed to stay in memory (12 bytes per pixel each) while a
/// later pass fuses. The encoded files, if the caller holds them, are not
/// included. All arithmetic is `u64` (a 32-bit `usize` overflows at 4 GiB).
#[must_use]
pub fn estimate_peak_bytes(
    frames: usize,
    width: u32,
    height: u32,
    settings: &StackingSettings,
    passes: &[PassKind],
) -> u64 {
    let n = u64::try_from(frames).unwrap_or(u64::MAX);
    if n == 0 {
        return 0;
    }
    let (w, h) = preprocessed_dimensions(width, height, &settings.preprocessing);
    let pixels = u64::from(w).saturating_mul(u64::from(h));
    let source_pixels = u64::from(width).saturating_mul(u64::from(height));
    // `n * per_frame + working`, all per pixel, times the pixels.
    let stage = |per_frame: u64, working: u64| {
        n.saturating_mul(per_frame)
            .saturating_add(working)
            .saturating_mul(pixels)
    };

    let mut peak = stage(FRAME_BYTES, DECODE_WORKING_BYTES)
        .saturating_add(DECODE_SOURCE_BYTES.saturating_mul(source_pixels));
    if settings.alignment_mode != AlignmentModeSetting::None && n > 1 {
        let mut working = ALIGN_WORKING_BYTES;
        if settings.chain_alignment {
            working += CHAIN_WORKING_BYTES;
        }
        let window = match (usize::try_from(w), usize::try_from(h)) {
            (Ok(w), Ok(h)) => lookahead_bytes(w, h, settings, false),
            _ => 0,
        };
        peak = peak.max(stage(FRAME_BYTES, working).saturating_add(window));
    }
    if settings.auto_cull {
        peak = peak.max(stage(CULL_FRAME_BYTES, CULL_WORKING_BYTES));
    }
    for (index, pass) in passes.iter().enumerate() {
        let (per_frame, working) = match pass {
            PassKind::Apex => (
                APEX_FRAME_BYTES,
                u64::try_from(apex_fold_bytes_per_pixel(settings.pyramid_levels as usize))
                    .unwrap_or(u64::MAX),
            ),
            PassKind::Relief if settings.relief_use_multigrid => (
                RELIEF_FRAME_BYTES,
                RELIEF_WORKING_BYTES + MULTIGRID_EXTRA_BYTES,
            ),
            PassKind::Relief => (RELIEF_FRAME_BYTES, RELIEF_WORKING_BYTES),
            // Two prepared frames and the accumulators are in the 150 bytes.
            PassKind::Strata => (STRATA_FRAME_BYTES, STRATA_WORKING_BYTES),
        };
        let earlier = u64::try_from(index)
            .unwrap_or(u64::MAX)
            .saturating_mul(FRAME_BYTES);
        peak = peak.max(stage(per_frame, working.saturating_add(earlier)));
    }
    peak
}

/// The largest Pre-resize percentage whose [`estimate_peak_bytes`] fits `budget`.
///
/// It steps down from the setting in multiples of 5 and never goes above the
/// setting. The setting itself is returned when it already fits. `None` when even
/// [`MIN_PRE_RESIZE_PERCENT`] (the lowest value the settings allow) does not.
#[must_use]
pub fn fit_pre_resize(
    frames: usize,
    width: u32,
    height: u32,
    settings: &StackingSettings,
    passes: &[PassKind],
    budget: u64,
) -> Option<u32> {
    let current = settings
        .preprocessing
        .pre_resize_percent
        .clamp(MIN_PRE_RESIZE_PERCENT, 100);
    let mut trial = settings.clone();
    let mut fits = |percent: u32| {
        trial.preprocessing.pre_resize_percent = percent;
        estimate_peak_bytes(frames, width, height, &trial, passes) <= budget
    };
    if fits(current) {
        return Some(current);
    }
    (MIN_PRE_RESIZE_PERCENT.div_ceil(PRE_RESIZE_STEP)..=current / PRE_RESIZE_STEP)
        .rev()
        .map(|step| step * PRE_RESIZE_STEP)
        .filter(|&percent| percent < current)
        .find(|&percent| fits(percent))
}
