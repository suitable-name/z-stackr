//! The rectangle that is valid in every aligned frame.
//!
//! Warping a frame onto the reference leaves a border that has no data (or, for
//! the edge-clamped warp, replicated data). Fusing and saving the full frame
//! would show that border, so the result is cropped to the largest axis-aligned
//! rectangle covered by every frame. A rectangle below half of the frame area is
//! refused (the output is then not cropped at all) and the frame responsible is
//! named: one wandering frame must not collapse the whole result.

use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use stacker_align::transform::{intersect_coverage, largest_true_rectangle};
use stacker_core::{fma::fma_f32, image::PlanarImage};

/// `(x, y, width, height)` in pixels.
pub type Rect = (usize, usize, usize, usize);

/// A crop smaller than this fraction of the frame area is refused.
pub const MIN_CROP_AREA_FRACTION: f64 = 0.5;

/// The largest rectangle that is `true` in every one of `masks` (each
/// `width × height`, row-major), or `None` when there is none or `masks` is
/// empty.
#[must_use]
pub fn common_rect<'a>(
    masks: impl IntoIterator<Item = &'a [bool]>,
    width: usize,
    height: usize,
) -> Option<Rect> {
    let mut common = CommonCoverage::new(width, height);
    let mut any = false;
    for (i, mask) in masks.into_iter().enumerate() {
        common.fold(i, mask);
        any = true;
    }
    if any { common.rect() } else { None }
}

/// The running intersection of the coverage masks of a stack, plus the frame
/// that covers the least (the likely culprit of a bad crop).
#[derive(Debug, Clone)]
pub struct CommonCoverage {
    width: usize,
    height: usize,
    mask: Vec<bool>,
    worst: Option<(usize, usize)>,
}

impl CommonCoverage {
    /// Starts all-covered: the reference frame is the identity and fully valid.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            mask: vec![true; width * height],
            worst: None,
        }
    }

    /// Fold the coverage of frame `frame_idx` into the intersection.
    ///
    /// # Panics
    /// Panics when `coverage` is not `width × height`.
    pub fn fold(&mut self, frame_idx: usize, coverage: &[bool]) {
        let covered = coverage.iter().filter(|&&c| c).count();
        if self.worst.is_none_or(|(_, least)| covered < least) {
            self.worst = Some((frame_idx, covered));
        }
        intersect_coverage(&mut self.mask, coverage);
    }

    /// The pixels covered by every frame folded so far (row-major).
    #[must_use]
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// The frame (index, covered pixels) with the smallest coverage.
    #[must_use]
    pub const fn worst(&self) -> Option<(usize, usize)> {
        self.worst
    }

    /// Pixels in a full frame.
    #[must_use]
    pub const fn frame_pixels(&self) -> usize {
        self.width * self.height
    }

    /// The largest rectangle covered by every frame so far, with no area test.
    #[must_use]
    pub fn rect(&self) -> Option<Rect> {
        largest_true_rectangle(&self.mask, self.width, self.height)
    }

    /// The crop to apply: the common rectangle, or the full frame when there is
    /// none or it is smaller than [`MIN_CROP_AREA_FRACTION`] of the frame. The
    /// refusal is logged with `name_of(frame)` naming the frame that covers the
    /// least.
    #[must_use]
    pub fn crop(&self, name_of: impl Fn(usize) -> String) -> CropDecision {
        let full = (0, 0, self.width, self.height);
        let Some(rect) = self.rect() else {
            tracing::warn!("no region is covered by every frame; the output is not cropped");
            return CropDecision {
                rect: full,
                refused: true,
            };
        };
        if exceeds_half(rect, self.frame_pixels()) {
            return CropDecision {
                rect,
                refused: false,
            };
        }
        let culprit = self.worst.map_or_else(String::new, |(idx, covered)| {
            format!(
                "; frame '{}' covers only {}% of the reference",
                name_of(idx),
                covered * 100 / self.frame_pixels().max(1)
            )
        });
        tracing::warn!(
            crop_w = rect.2,
            crop_h = rect.3,
            "the common valid area is below 50% of the frame; refusing to crop{culprit}"
        );
        CropDecision {
            rect: full,
            refused: true,
        }
    }
}

impl CommonCoverage {
    /// [`Self::crop`] honouring the `crop_to_common_area` setting: with
    /// `enabled == false` the whole frame is kept (no crop, nothing refused, no
    /// warning), and the border the alignment left shows in the result.
    #[must_use]
    pub fn crop_when(&self, enabled: bool, name_of: impl Fn(usize) -> String) -> CropDecision {
        if enabled {
            self.crop(name_of)
        } else {
            CropDecision {
                rect: (0, 0, self.width, self.height),
                refused: false,
            }
        }
    }
}

/// `true` when `rect` covers at least [`MIN_CROP_AREA_FRACTION`] of `frame_pixels`.
const fn exceeds_half(rect: Rect, frame_pixels: usize) -> bool {
    rect.2 * rect.3 * 2 >= frame_pixels
}

/// Outcome of [`CommonCoverage::crop`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropDecision {
    /// The rectangle to keep; the full frame when the crop was refused.
    pub rect: Rect,
    /// `true` when no usable rectangle existed (logged as a warning).
    pub refused: bool,
}

impl CropDecision {
    /// `Some(rect)` when the crop actually removes something, `None` for the
    /// full frame.
    #[must_use]
    pub fn needed(&self, width: usize, height: usize) -> Option<Rect> {
        (self.rect != (0, 0, width, height)).then_some(self.rect)
    }
}

// ── Resizing the cropped result back (`resize_cropped_to_original`) ───────────

#[expect(
    clippy::cast_precision_loss,
    reason = "image edges are far below 2^52, so the usize to f64 conversion is exact"
)]
const fn dim_f64(n: usize) -> f64 {
    n as f64
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the weights are narrowed to the f32 the planes hold"
)]
const fn narrow(x: f64) -> f32 {
    x as f32
}

/// For every output index of an axis `src → dst` the four source indices of a
/// Catmull-Rom kernel (edge-clamped) and their weights. The output pixel `o`
/// samples the source at `(o + ½)·src/dst − ½` (pixel centres).
fn axis_taps(src: usize, dst: usize) -> Vec<([usize; 4], [f32; 4])> {
    let scale = dim_f64(src) / dim_f64(dst);
    let last = isize::try_from(src - 1).unwrap_or(isize::MAX);
    (0..dst)
        .map(|o| {
            #[expect(
                clippy::suboptimal_flops,
                reason = "an unfused product keeps the sample positions identical on every target"
            )]
            let position = (dim_f64(o) + 0.5) * scale - 0.5;
            let base = position.floor();
            let t = narrow(position - base);
            let base = float_to_isize(base);
            let index = |k: isize| usize::try_from((base + k).clamp(0, last)).unwrap_or(0);
            let w0 = fma_f32(fma_f32(-0.5, t, 1.0), t, -0.5) * t;
            let w1 = fma_f32(fma_f32(1.5, t, -2.5) * t, t, 1.0);
            let w2 = fma_f32(fma_f32(-1.5, t, 2.0), t, 0.5) * t;
            let w3 = fma_f32(0.5, t, -0.5) * t * t;
            ([index(-1), index(0), index(1), index(2)], [w0, w1, w2, w3])
        })
        .collect()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the argument is a floored pixel position, far inside isize"
)]
const fn float_to_isize(x: f64) -> isize {
    x as isize
}

/// The four-tap weighted sum of `taps` over `value(index)`.
fn sum4(taps: &([usize; 4], [f32; 4]), value: impl Fn(usize) -> f32) -> f32 {
    let (index, weight) = taps;
    fma_f32(
        weight[3],
        value(index[3]),
        fma_f32(
            weight[2],
            value(index[2]),
            fma_f32(weight[1], value(index[1]), weight[0] * value(index[0])),
        ),
    )
}

/// Resample one plane with the separable Catmull-Rom kernel.
///
/// `src` is `sw × sh` and every dimension is non-zero.
pub(crate) fn resize_plane(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let xs = axis_taps(sw, dw);
    let ys = axis_taps(sh, dh);

    // Horizontal pass: `sh` rows of `dw` samples.
    let mut tmp = vec![0.0_f32; dw * sh];
    tmp.par_chunks_mut(dw)
        .zip(src.par_chunks(sw))
        .for_each(|(out, row)| {
            for (cell, taps) in out.iter_mut().zip(&xs) {
                *cell = sum4(taps, |i| row[i]);
            }
        });

    // Vertical pass.
    let mut out = vec![0.0_f32; dw * dh];
    out.par_chunks_mut(dw)
        .zip(ys.par_iter())
        .for_each(|(row_out, taps)| {
            for (x, cell) in row_out.iter_mut().enumerate() {
                *cell = sum4(taps, |i| tmp[i * dw + x]);
            }
        });
    out
}

/// Resample `image` to `width × height` with an edge-clamped Catmull-Rom spline
/// (pixel centres at `(i + ½)·scale − ½`), plane by plane.
///
/// This is `resize_cropped_to_original`: after a common-area crop the result is
/// stretched back to the canvas it came from. The stretch is slightly non-uniform
/// whenever the crop's aspect ratio differs from the canvas's. An empty source or
/// target gives an all-zero image of the target size; an image that already has
/// the target size is returned as it is.
#[must_use]
pub fn resize_planar(image: PlanarImage<f32>, width: usize, height: usize) -> PlanarImage<f32> {
    if (image.width, image.height) == (width, height) {
        return image;
    }
    let mut out = PlanarImage::new(width, height);
    if image.width == 0 || image.height == 0 || width == 0 || height == 0 {
        return out;
    }
    let (sw, sh) = (image.width, image.height);
    out.luma = resize_plane(&image.luma, sw, sh, width, height);
    out.chroma_a = resize_plane(&image.chroma_a, sw, sh, width, height);
    out.chroma_b = resize_plane(&image.chroma_b, sw, sh, width, height);
    out
}
