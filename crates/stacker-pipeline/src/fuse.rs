//! Fusing a stack: `Apex` (Laplacian pyramid), `Relief` (focus map, contrast
//! mask, per-pixel selection, guided smoothing or a multigrid depth map) and
//! `Strata` (guided-filter soft blend).
//!
//! The functions take a slice of equally sized frames. The CLI calls them once
//! per padded tile, the desktop app once for the whole image; because every
//! setting is read here, in one place, the two cannot disagree on what a key
//! means.
//!
//! # The `Relief` threshold belongs to the whole image
//!
//! `relief_contrast_pct` (a rejected fraction) and `relief_auto_detect` (the
//! median + MAD noise-floor rule) are evaluated on the **whole-image** focus map,
//! once: [`relief_threshold`] turns them into an absolute focus value. A tiled
//! caller computes it from the whole-image map (the per-pixel maximum of the
//! frames' focus maps) and hands the value to [`relief_tile`] for every tile;
//! evaluating the percentile or the auto-detection per tile would give each tile
//! its own threshold and the mask would jump at the tile borders. [`relief`] does
//! both steps for a whole image.
//!
//! # `Relief` with the multigrid solver
//!
//! With `relief_use_multigrid` the per-pixel winner is not smoothed by a guided
//! filter but solved into a continuous depth map. The solver works on
//! **whole-image** planes only: the best focus value and the winning frame index
//! of every pixel ([`DepthVotes`], 6 bytes per pixel). A tiled caller folds the
//! interior of every tile into them ([`DepthVotes::merge_window`]), solves once
//! ([`DepthVotes::depth`]) and composites every tile with the matching window of
//! the depth ([`relief_multigrid_tile`]), so the result cannot show seams. The
//! guided path stays the default and is untouched.
//!
//! # `Strata`
//!
//! [`strata`] runs the CPU kernels; [`strata_on`] takes the kernels and the frame
//! source from the caller (the device seam: a GPU implementation of
//! [`StrataKernels`] plugs in there).

use rayon::prelude::*;
use stacker_algo::{
    apex::fuse::fuse_streaming,
    relief::{
        focus::compute_focus_map,
        guided::guided_filter_luma,
        multigrid::{composite_depth, confidence_weights, multigrid_depth, update_argmax},
        threshold::{ReliefThreshold, compute_relief_threshold, mask_above},
    },
    strata::{
        FrameSource as StrataFrames, MAX_BASE_RADIUS, StrataKernels, StrataParams, fuse_strata,
        fuse_strata_on,
    },
};
use stacker_core::{image::PlanarImage, settings::StackingSettings};

use crate::compute::Compute;

/// Regularisation of the guided filter that smooths the selected luma.
const GUIDED_EPS: f32 = 0.01;

/// `Apex` fusion of `images`: Laplacian pyramids, per-level maximum contrast
/// selection, reconstruction. Reads `pyramid_levels`, `use_all_color_channels`
/// and `grit_suppression`.
///
/// The frames are folded in one at a time into a single accumulator pyramid
/// (`stacker_algo::apex::fuse::ApexFold`), so the memory beyond the frames
/// themselves is `apex_fold_bytes_per_pixel(levels)` (84 B/px at 8 levels) whatever
/// the frame count. The result equals the batch fusion of all pyramids
/// (`build_and_fuse_pyramids` followed by `reconstruct`) bit for bit.
#[must_use]
pub fn apex(images: &[PlanarImage<f32>], settings: &StackingSettings) -> PlanarImage<f32> {
    if images.is_empty() {
        return PlanarImage::new(0, 0);
    }
    fuse_streaming(
        images,
        settings.pyramid_levels as usize,
        settings.use_all_color_channels,
        settings.grit_suppression,
        |_, _| {},
    )
}

/// The focus measure of every frame and their per-pixel maximum.
#[derive(Debug)]
pub struct FocusMaps {
    /// One single-plane focus map per frame (`width × height` samples each).
    pub smls: Vec<Vec<f32>>,
    /// The best focus any frame reaches at each pixel (the `luma` plane holds
    /// the measure; the chroma planes are zero).
    pub max: PlanarImage<f32>,
}

/// Compute the focus map of every frame (in parallel) with the given
/// estimation radius (`relief_estimation_radius`).
///
/// # Panics
/// Panics when `images` is empty.
#[must_use]
pub fn focus_maps(images: &[PlanarImage<f32>], estimation_radius: usize) -> FocusMaps {
    let smls: Vec<Vec<f32>> = images
        .par_iter()
        .map(|img| compute_focus_map(img, estimation_radius))
        .collect();
    let (width, height) = (images[0].width, images[0].height);
    let mut max_luma = vec![f32::NEG_INFINITY; width * height];
    for sml in &smls {
        for (m, &s) in max_luma.iter_mut().zip(sml) {
            if s > *m {
                *m = s;
            }
        }
    }
    let max = PlanarImage::with_luma(width, height, max_luma)
        .expect("the maximum has one sample per pixel");
    FocusMaps { smls, max }
}

/// The absolute `Relief` threshold for a **whole-image** focus map.
///
/// It is the `relief_contrast_pct` percentile, or the median + MAD noise-floor estimate
/// (`stacker_algo::relief::threshold::auto_detect_threshold`) with
/// `relief_auto_detect`. See the module documentation for why this is computed
/// once.
#[must_use]
pub fn relief_threshold(whole_focus_map: &[f32], settings: &StackingSettings) -> ReliefThreshold {
    compute_relief_threshold(
        whole_focus_map,
        settings.relief_contrast_pct,
        settings.relief_auto_detect,
    )
}

/// Per-pixel best-source selection with the contrast-mask fallback, then
/// edge-preserving guided-filter smoothing of the luma.
///
/// Where `mask[i]` is `true` the pixel comes from the frame with the highest
/// focus value (ties go to the earlier frame); elsewhere it is the mean of the
/// frames. The chroma planes are not smoothed.
///
/// # Panics
/// Panics when `images` is empty or `smls` / `mask` do not match it in length.
#[must_use]
pub fn select_and_smooth(
    images: &[PlanarImage<f32>],
    smls: &[Vec<f32>],
    mask: &[bool],
    smoothing_radius: usize,
) -> PlanarImage<f32> {
    let (width, height) = (images[0].width, images[0].height);
    let len = width * height;
    assert_eq!(smls.len(), images.len(), "one focus map per frame");
    assert_eq!(mask.len(), len, "one mask entry per pixel");
    #[expect(
        clippy::cast_precision_loss,
        reason = "a frame count is far below 2^24"
    )]
    let n_frames = images.len() as f32;
    let mut out_luma = vec![0.0_f32; len];
    let mut out_chroma_a = vec![0.0_f32; len];
    let mut out_chroma_b = vec![0.0_f32; len];

    out_luma
        .par_iter_mut()
        .zip(out_chroma_a.par_iter_mut())
        .zip(out_chroma_b.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((ol, oa), ob))| {
            if mask[i] {
                // Above the contrast threshold: the sharpest frame wins.
                let mut best = f32::NEG_INFINITY;
                for (img, sml) in images.iter().zip(smls) {
                    if sml[i] > best {
                        best = sml[i];
                        *ol = img.luma[i];
                        *oa = img.chroma_a[i];
                        *ob = img.chroma_b[i];
                    }
                }
            } else {
                // Below it (flat area): the mean of all frames avoids artefacts.
                let mut sum_l = 0.0_f32;
                let mut sum_a = 0.0_f32;
                let mut sum_b = 0.0_f32;
                for img in images {
                    sum_l += img.luma[i];
                    sum_a += img.chroma_a[i];
                    sum_b += img.chroma_b[i];
                }
                *ol = sum_l / n_frames;
                *oa = sum_a / n_frames;
                *ob = sum_b / n_frames;
            }
        });

    smooth_selected(
        width,
        height,
        &out_luma,
        out_chroma_a,
        out_chroma_b,
        smoothing_radius,
    )
}

/// The luma guides and is smoothed; the chroma planes pass through.
fn smooth_selected(
    width: usize,
    height: usize,
    luma: &[f32],
    chroma_a: Vec<f32>,
    chroma_b: Vec<f32>,
    smoothing_radius: usize,
) -> PlanarImage<f32> {
    let smoothed = guided_filter_luma(luma, luma, width, height, smoothing_radius, GUIDED_EPS);
    PlanarImage::from_planes(width, height, smoothed, chroma_a, chroma_b)
        .expect("every plane has one sample per pixel")
}

/// The focus maps of `images` folded into the best focus value and the winning
/// frame of every pixel.
///
/// One frame is folded at a time (6 B/px kept, plus one focus map in
/// flight), instead of one focus map per frame ([`focus_maps`], 4 B/px *per
/// frame*). The earlier frame wins a tie. Callers with more than
/// [`MAX_MULTIGRID_FRAMES`] frames use [`focus_maps`].
///
/// # Panics
/// Panics when there are more than [`MAX_MULTIGRID_FRAMES`] frames.
#[must_use]
pub fn focus_votes(images: &[PlanarImage<f32>], estimation_radius: usize) -> DepthVotes {
    multigrid_votes(images, estimation_radius)
}

/// The best focus any frame reaches at each pixel (`width * height` values): the
/// maximum focus map, as `focus_maps(images, radius).max.luma`, without keeping a
/// map per frame.
///
/// # Panics
/// Panics when `images` is empty.
#[must_use]
pub fn focus_max_map(images: &[PlanarImage<f32>], estimation_radius: usize) -> Vec<f32> {
    if images.len() > MAX_MULTIGRID_FRAMES {
        return focus_maps(images, estimation_radius).max.luma;
    }
    assert!(!images.is_empty(), "there is no frame to measure");
    focus_votes(images, estimation_radius).into_parts().0
}

/// [`select_and_smooth`] from the streamed votes instead of one focus map per
/// frame.
///
/// Where `best[i] >= threshold` the pixel comes from frame `index[i]` (the
/// first frame with the highest focus value), elsewhere it is the mean of the
/// frames. The result equals `select_and_smooth(images, &focus_maps(..).smls,
/// &mask_above(best, threshold), smoothing_radius)` bit for bit.
///
/// # Panics
/// Panics when `images` is empty or `best` / `index` do not hold one value per
/// pixel.
#[must_use]
pub fn select_from_votes(
    images: &[PlanarImage<f32>],
    best: &[f32],
    index: &[u16],
    threshold: f32,
    smoothing_radius: usize,
) -> PlanarImage<f32> {
    let (width, height) = (images[0].width, images[0].height);
    let len = width * height;
    assert_eq!(best.len(), len, "one best-focus value per pixel");
    assert_eq!(index.len(), len, "one winning frame per pixel");
    #[expect(
        clippy::cast_precision_loss,
        reason = "a frame count is far below 2^24"
    )]
    let n_frames = images.len() as f32;
    let mut out_luma = vec![0.0_f32; len];
    let mut out_chroma_a = vec![0.0_f32; len];
    let mut out_chroma_b = vec![0.0_f32; len];

    out_luma
        .par_iter_mut()
        .zip(out_chroma_a.par_iter_mut())
        .zip(out_chroma_b.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((ol, oa), ob))| {
            if best[i] >= threshold {
                // Above the contrast threshold: the sharpest frame wins. A pixel
                // whose focus values never rose above -inf has no winner and stays
                // 0, as in the per-frame-map path.
                if best[i] > f32::NEG_INFINITY {
                    let winner = &images[usize::from(index[i])];
                    *ol = winner.luma[i];
                    *oa = winner.chroma_a[i];
                    *ob = winner.chroma_b[i];
                }
            } else {
                // Below it (flat area): the mean of all frames avoids artefacts.
                let mut sum_l = 0.0_f32;
                let mut sum_a = 0.0_f32;
                let mut sum_b = 0.0_f32;
                for img in images {
                    sum_l += img.luma[i];
                    sum_a += img.chroma_a[i];
                    sum_b += img.chroma_b[i];
                }
                *ol = sum_l / n_frames;
                *oa = sum_a / n_frames;
                *ob = sum_b / n_frames;
            }
        });

    smooth_selected(
        width,
        height,
        &out_luma,
        out_chroma_a,
        out_chroma_b,
        smoothing_radius,
    )
}

/// `Relief` fusion of a whole image.
///
/// Focus maps, the threshold from the maximum map ([`relief_threshold`]),
/// selection and smoothing. Reads `relief_estimation_radius`,
/// `relief_smoothing_radius`, `relief_contrast_pct`, `relief_auto_detect` and
/// `relief_use_multigrid` (on: [`relief_multigrid`] instead of the guided
/// smoothing).
///
/// The focus maps are folded into the best value and the winning frame of every
/// pixel as they are computed ([`focus_votes`]), so the memory beyond the frames
/// is 6 B/px plus one focus map, not 4 B/px per frame. The result equals the one
/// built from all the maps ([`focus_maps`], [`select_and_smooth`]) bit for bit.
#[must_use]
pub fn relief(images: &[PlanarImage<f32>], settings: &StackingSettings) -> PlanarImage<f32> {
    if images.is_empty() {
        return PlanarImage::new(0, 0);
    }
    if settings.relief_use_multigrid {
        return relief_multigrid(images, settings);
    }
    let radius = settings.relief_estimation_radius as usize;
    let smoothing = settings.relief_smoothing_radius as usize;
    if images.len() > MAX_MULTIGRID_FRAMES {
        // More frames than a `u16` winner index holds: one map per frame.
        let maps = focus_maps(images, radius);
        let threshold = relief_threshold(&maps.max.luma, settings);
        let mask = mask_above(&maps.max.luma, threshold.value);
        return select_and_smooth(images, &maps.smls, &mask, smoothing);
    }
    let votes = focus_votes(images, radius);
    let threshold = votes.threshold(settings);
    select_from_votes(
        images,
        votes.best(),
        votes.index(),
        threshold.value,
        smoothing,
    )
}

/// `Relief` fusion of one padded tile with the **whole-image** `threshold`
/// (the `value` of [`relief_threshold`] on the whole-image focus map).
///
/// This is always the guided path. With `relief_use_multigrid` a tile cannot be
/// fused from a threshold alone: use [`DepthVotes`] and [`relief_multigrid_tile`].
/// Streams the focus maps like [`relief`].
#[must_use]
pub fn relief_tile(
    images: &[PlanarImage<f32>],
    settings: &StackingSettings,
    threshold: f32,
) -> PlanarImage<f32> {
    if images.is_empty() {
        return PlanarImage::new(0, 0);
    }
    let radius = settings.relief_estimation_radius as usize;
    let smoothing = settings.relief_smoothing_radius as usize;
    if images.len() > MAX_MULTIGRID_FRAMES {
        let maps = focus_maps(images, radius);
        let mask = mask_above(&maps.max.luma, threshold);
        return select_and_smooth(images, &maps.smls, &mask, smoothing);
    }
    let votes = focus_votes(images, radius);
    select_from_votes(images, votes.best(), votes.index(), threshold, smoothing)
}

// ── Relief with the multigrid solver ────────────────────────────────────────

/// The most frames a multigrid depth map can index: the winning frame is kept as
/// a `u16`.
pub const MAX_MULTIGRID_FRAMES: usize = 1 << 16;

/// The whole-image planes the multigrid solver needs: for every pixel the best
/// focus value any frame reaches and the index of the frame that reaches it.
///
/// 6 bytes per pixel. Build it frame by frame ([`Self::fold_frame`]) or, for a
/// tiled run, tile by tile ([`Self::merge_window`]); both give the planes of a
/// whole-image run. The earlier frame wins a tie (strict `>`), and a pixel whose
/// focus values are all NaN keeps frame 0.
#[derive(Debug, Clone)]
pub struct DepthVotes {
    width: usize,
    height: usize,
    best: Vec<f32>,
    index: Vec<u16>,
}

impl DepthVotes {
    /// Empty votes for a `width × height` image (no pixel has a vote yet).
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            best: vec![f32::NEG_INFINITY; width * height],
            index: vec![0; width * height],
        }
    }

    /// The planes `(best focus, winning frame)`, moved out.
    #[must_use]
    pub fn into_parts(self) -> (Vec<f32>, Vec<u16>) {
        (self.best, self.index)
    }

    /// Width in pixels.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// The best focus value per pixel: the maximum focus map, from which the
    /// contrast threshold is computed.
    #[must_use]
    pub fn best(&self) -> &[f32] {
        &self.best
    }

    /// The index of the winning frame per pixel.
    #[must_use]
    pub fn index(&self) -> &[u16] {
        &self.index
    }

    /// Offer the focus map `focus` of frame number `frame` (its position in the
    /// stack, frames in stack order).
    ///
    /// # Panics
    /// Panics when `frame` is not below [`MAX_MULTIGRID_FRAMES`] or `focus` does
    /// not hold one value per pixel.
    pub fn fold_frame(&mut self, frame: usize, focus: &[f32]) {
        let frame = u16::try_from(frame)
            .expect("a multigrid depth map indexes at most MAX_MULTIGRID_FRAMES frames");
        update_argmax(&mut self.best, &mut self.index, frame, focus);
    }

    /// Copy the rectangle `dest = (x, y, width, height)` of this image from the
    /// same-sized window of `local`, whose top-left corner for it is `local_offset`:
    /// the interior of a tile's votes (computed on the padded tile) into the
    /// whole-image votes.
    ///
    /// # Panics
    /// Panics when the window reaches outside `local` or `dest` outside `self`.
    pub fn merge_window(
        &mut self,
        local: &Self,
        local_offset: (usize, usize),
        dest: (usize, usize, usize, usize),
    ) {
        let (ix, iy) = local_offset;
        let (dx, dy, w, h) = dest;
        assert!(
            ix + w <= local.width && iy + h <= local.height,
            "the window reaches outside the local votes"
        );
        assert!(
            dx + w <= self.width && dy + h <= self.height,
            "the destination reaches outside the image"
        );
        for row in 0..h {
            let src = (iy + row) * local.width + ix;
            let at = (dy + row) * self.width + dx;
            self.best[at..at + w].copy_from_slice(&local.best[src..src + w]);
            self.index[at..at + w].copy_from_slice(&local.index[src..src + w]);
        }
    }

    /// The `Relief` contrast threshold of the maximum focus map, by the same rule
    /// the guided path uses ([`relief_threshold`]).
    #[must_use]
    pub fn threshold(&self, settings: &StackingSettings) -> ReliefThreshold {
        relief_threshold(&self.best, settings)
    }

    /// The whole-image depth map: pixels whose best focus reaches `threshold` are
    /// pinned to their winning frame, the rest is diffused from them, then
    /// smoothed over `relief_smoothing_radius` octaves (see
    /// `stacker_algo::relief::multigrid::multigrid_depth`). The solver peaks at
    /// about 28 bytes per pixel; the result is 4.
    #[must_use]
    pub fn depth(&self, threshold: f32, settings: &StackingSettings) -> Vec<f32> {
        multigrid_depth_map(
            (self.width, self.height),
            &self.best,
            &self.index,
            threshold,
            settings,
        )
    }

    /// [`Self::depth`] through `compute`: the GPU solver when the handle has a
    /// backend that serves it, the CPU solver otherwise and whenever the device
    /// cannot deliver (see [`Compute::multigrid_depth`]).
    #[must_use]
    pub fn depth_on(
        &self,
        compute: &mut Compute,
        threshold: f32,
        settings: &StackingSettings,
    ) -> Vec<f32> {
        multigrid_depth_map_on(
            compute,
            (self.width, self.height),
            &self.best,
            &self.index,
            threshold,
            settings,
        )
    }
}

/// [`multigrid_depth_map`] through `compute` (the GPU solver with the CPU solver
/// behind it, see [`Compute::multigrid_depth`]).
///
/// # Panics
/// Panics when a plane holds fewer than `width * height` values.
#[must_use]
pub fn multigrid_depth_map_on(
    compute: &mut Compute,
    (width, height): (usize, usize),
    best: &[f32],
    index: &[u16],
    threshold: f32,
    settings: &StackingSettings,
) -> Vec<f32> {
    let weights = confidence_weights(best, threshold);
    compute.multigrid_depth(
        width,
        height,
        index,
        &weights,
        settings.relief_smoothing_radius as usize,
    )
}

/// The whole-image depth map of the planes `best` (best focus per pixel) and
/// `index` (winning frame per pixel) of a `(width, height)` image, as
/// [`DepthVotes::depth`] solves it.
///
/// # Panics
/// Panics when a plane holds fewer than `width * height` values.
#[must_use]
pub fn multigrid_depth_map(
    (width, height): (usize, usize),
    best: &[f32],
    index: &[u16],
    threshold: f32,
    settings: &StackingSettings,
) -> Vec<f32> {
    let weights = confidence_weights(best, threshold);
    multigrid_depth(
        width,
        height,
        index,
        &weights,
        settings.relief_smoothing_radius as usize,
    )
}

/// The votes of `images` (frames in stack order, all the same size): the focus map
/// of every frame is computed one after the other and folded in, so only one
/// frame's map is alive at a time.
///
/// # Panics
/// Panics when there are more than [`MAX_MULTIGRID_FRAMES`] frames.
#[must_use]
pub fn multigrid_votes(images: &[PlanarImage<f32>], estimation_radius: usize) -> DepthVotes {
    let Some(first) = images.first() else {
        return DepthVotes::new(0, 0);
    };
    let mut votes = DepthVotes::new(first.width, first.height);
    for (frame, image) in images.iter().enumerate() {
        votes.fold_frame(frame, &compute_focus_map(image, estimation_radius));
    }
    votes
}

/// `Relief` fusion of a whole image with the multigrid depth map.
///
/// The steps are the votes, the threshold from the best focus
/// ([`relief_threshold`]), the solved depth, and the per-pixel interpolation
/// between the neighbouring frames.
///
/// The frame order matters: the depth is a position in `images`, so the frames
/// should be in capture (focus) order.
///
/// # Panics
/// Panics when there are more than [`MAX_MULTIGRID_FRAMES`] frames.
#[must_use]
pub fn relief_multigrid(
    images: &[PlanarImage<f32>],
    settings: &StackingSettings,
) -> PlanarImage<f32> {
    if images.is_empty() {
        return PlanarImage::new(0, 0);
    }
    let votes = multigrid_votes(images, settings.relief_estimation_radius as usize);
    let threshold = votes.threshold(settings);
    let depth = votes.depth(threshold.value, settings);
    relief_multigrid_tile(images, &depth)
}

/// The fused image of `images` (a whole image or one padded tile) for `depth`.
///
/// `depth` is the matching window of the whole-image depth (cut with
/// `stacker_algo::relief::multigrid::crop_plane`). Per pixel, so a tile of the
/// whole-image result is what this returns for the tile's frames.
///
/// # Panics
/// Panics when `images` differ in size or `depth` does not hold one value per pixel.
#[must_use]
pub fn relief_multigrid_tile(images: &[PlanarImage<f32>], depth: &[f32]) -> PlanarImage<f32> {
    composite_depth(images, depth)
}

// ── Strata ──────────────────────────────────────────────────────────────────

/// A boxed set of `Strata` kernels (what [`Compute::strata_kernels`] hands out)
/// as a [`StrataKernels`] value of its own, so it can be passed where a
/// `&mut impl StrataKernels` is wanted (`strata_on`, `fuse_strata_with`).
pub struct BoxedStrata(pub Box<dyn StrataKernels>);

impl std::fmt::Debug for BoxedStrata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoxedStrata").finish_non_exhaustive()
    }
}

impl StrataKernels for BoxedStrata {
    fn saliency(&mut self, luma: &[f32], width: usize, height: usize) -> Option<Vec<f32>> {
        self.0.saliency(luma, width, height)
    }

    fn guided_pair(
        &mut self,
        guidance: &[f32],
        src: &[f32],
        width: usize,
        height: usize,
        radius_a: usize,
        eps_a: f32,
        radius_b: usize,
        eps_b: f32,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        self.0.guided_pair(
            guidance, src, width, height, radius_a, eps_a, radius_b, eps_b,
        )
    }
}

/// The smallest `strata_base_radius` the settings accept.
pub const MIN_STRATA_BASE_RADIUS: usize = 8;

/// The largest `strata_detail_focus` the settings accept.
pub const MAX_STRATA_DETAIL_FOCUS: u32 = 5;

/// The Strata parameters of `settings`, clamped to the ranges the settings allow.
///
/// They are `strata_base_radius` and `strata_detail_focus`, clamped to `8..=64`
/// and `1..=5` so a hand-built settings value cannot break the apron
/// assumption of [`crate::tiling::apron_for`].
#[must_use]
pub fn strata_params(settings: &StackingSettings) -> StrataParams {
    StrataParams {
        base_radius: (settings.strata_base_radius as usize)
            .clamp(MIN_STRATA_BASE_RADIUS, MAX_BASE_RADIUS),
        detail_focus: settings
            .strata_detail_focus
            .clamp(1, MAX_STRATA_DETAIL_FOCUS),
    }
}

/// `Strata` fusion of `images` on the CPU. Reads `strata_base_radius` and
/// `strata_detail_focus`.
///
/// # Panics
/// Panics when there are more than 65 536 frames or the frames differ in size.
#[must_use]
pub fn strata(images: &[PlanarImage<f32>], settings: &StackingSettings) -> PlanarImage<f32> {
    if images.is_empty() {
        return PlanarImage::new(0, 0);
    }
    fuse_strata(images, &strata_params(settings))
}

/// `Strata` fusion with the kernels and the frame source of the caller.
///
/// This is the seam for a device (a GPU `StrataKernels` implementation; what it declines runs on
/// the CPU) and for frames that are produced on demand (both passes ask for
/// every frame once, in order).
///
/// `on_tick` is called `2 × frames` times, `on_preview` as described in
/// `stacker_algo::strata::fuse_strata_on` (`preview_every = NO_PREVIEW` switches
/// previews off).
///
/// # Panics
/// Panics when there is no frame, more than 65 536 frames, or the frames differ
/// in size.
#[must_use]
pub fn strata_on(
    kernels: &mut impl StrataKernels,
    frames: &(impl StrataFrames + ?Sized),
    settings: &StackingSettings,
    on_tick: impl FnMut(),
    preview_every: usize,
    on_preview: impl FnMut(&PlanarImage<f32>),
) -> PlanarImage<f32> {
    fuse_strata_on(
        kernels,
        frames,
        &strata_params(settings),
        on_tick,
        preview_every,
        on_preview,
    )
}
