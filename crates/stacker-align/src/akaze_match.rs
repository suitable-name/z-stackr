//! AKAZE keypoint extraction and brute-force descriptor matching on the luma
//! plane of a [`PlanarImage`].
//!
//! Detection runs on a temporary 8-bit, gamma-encoded, contrast-stretched copy
//! of the luma ([`luma_to_gray8`]); the linear-light planes the pipeline fuses
//! are never touched. Matching is a Hamming-distance nearest-neighbour search
//! with Lowe's ratio test, parallel over the query descriptors with an order
//! preserving collect, so the result does not depend on scheduling.

use akaze::Akaze;
// Re-exported publicly because `KeyPoint` appears in this module's public API
// (e.g. `extract_ref_features` returns `Vec<KeyPoint>`), so callers must be able
// to name it via `stacker_align::akaze_match::KeyPoint`.
pub use akaze::KeyPoint;
use bitarray::BitArray;
use image::{DynamicImage, ImageBuffer, Luma};
use rayon::prelude::*;
use stacker_core::{color::linear_to_srgb, error::StackerError, image::PlanarImage};

/// Namespace for the keypoint matching entry points
/// ([`Self::extract_and_match`], [`Self::match_target`]).
pub struct KeypointMatcher;

/// Match structure, defined locally because `akaze` 0.7.0 removed its built-in `Match` type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Match {
    /// Index into the reference keypoints (and descriptors).
    pub index_0: usize,
    /// Index into the target keypoints (and descriptors).
    pub index_1: usize,
}

/// AKAZE 0.7.0 relies on the standard `image` crate. We use an 8-bit grayscale
/// buffer because `DynamicImage` does not support an `ImageLuma32F` variant.
pub type Gray8Image = ImageBuffer<Luma<u8>, Vec<u8>>;

/// M-LDB descriptors in AKAZE 0.7.0 are returned as `BitArray<64>`.
pub type Descriptor = BitArray<64>;

/// Result of feature extraction + matching: matched index pairs and the two
/// keypoint vectors needed to resolve them into coordinates.
pub struct MatchResult {
    /// The accepted matches, in ascending order of `index_0`.
    pub matches: Vec<Match>,
    /// Reference keypoints; `Match::index_0` indexes this.
    pub kps0: Vec<KeyPoint>,
    /// Target keypoints; `Match::index_1` indexes this.
    pub kps1: Vec<KeyPoint>,
}

// ── internal helpers ────────────────────────────────────────────────────────

/// Build an 8-bit `Gray8Image` from the linear-light luma plane of a
/// `PlanarImage<f32>`, applying two steps to restore perceptual contrast
/// for the AKAZE detector:
///
/// ## Why two steps are needed
///
/// The fusion pipeline stores luma in **scene-linear** light (after
/// `srgb_to_linear`).  Linear light compresses midtone contrast: a
/// perceptually mid-grey value (sRGB ≈ 0.50) sits at linear ≈ 0.214.
/// AKAZE's contrast-based detector uses Scharr derivatives and a
/// contrast-factor histogram computed on whatever buffer you feed it.  On
/// a low-amplitude linear buffer the derivative magnitudes are tiny and
/// the detector threshold filters out almost every candidate.
///
/// ## Step 1 — gamma re-encode (linear → perceptual sRGB)
///
/// Apply `stacker_core::color::linear_to_srgb` per pixel.  This is the
/// exact inverse of the `srgb_to_linear` applied during loading, so it
/// restores the midtone contrast the detector needs.
///
/// ## Step 2 — robust 1st–99th percentile contrast stretch
///
/// Even after gamma re-encoding a specific frame may use only a narrow
/// sub-range of `[0, 1]`. We sort a histogram of 256 bins to find the
/// pixel values at the 1st and 99th percentile, then linearly remap
/// `[p1, p99]` → `[0, 255]`. Using the 1st/99th percentiles makes the
/// stretch robust against isolated specular highlights or stuck pixels.
///
/// ## What is NOT changed
///
/// `PlanarImage::luma`, `chroma_a`, and `chroma_b` are **not** modified.
/// The returned 8-bit image is a temporary, detector-only buffer.
/// Fusion continues to see the original linear-light f32 values.
///
/// # Panics
/// Never in practice: the output buffer is built with exactly
/// `width * height` bytes, so wrapping it as an image cannot fail.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "image dimensions fit a u32; the histogram bin, percentile counts and 8-bit output are clamped or non-negative and exact to one pixel/bin (the f32 percentile is the long-standing detector input, kept bit-for-bit)"
)]
pub fn luma_to_gray8(img: &PlanarImage<f32>) -> Gray8Image {
    /// Pixels per rayon work item in the gamma/histogram pass.
    const CHUNK: usize = 1 << 16;

    let npx = img.width * img.height;
    let (width, height) = (img.width as u32, img.height as u32);

    // ── Step 1: gamma re-encode each pixel, histogramming as we go ──────────
    //
    // One parallel pass for the gamma encode and the histogram.  The histogram is
    // an integer sum, so combining the per-chunk histograms in any order gives
    // the same result as a serial pass.
    let mut gamma = vec![0.0_f32; npx];
    let hist: [u32; 256] = gamma
        .par_chunks_mut(CHUNK)
        .zip(img.luma.par_chunks(CHUNK))
        .fold(
            || [0u32; 256],
            |mut hist, (g_chunk, l_chunk)| {
                for (g, &l) in g_chunk.iter_mut().zip(l_chunk) {
                    let v = linear_to_srgb(l.clamp(0.0, 1.0));
                    *g = v;
                    let bin = (v.clamp(0.0, 1.0) * 255.0) as usize;
                    hist[bin.min(255)] += 1;
                }
                hist
            },
        )
        .reduce(
            || [0u32; 256],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );

    // ── Step 2: 1st–99th percentile contrast stretch ────────────────────────

    let thresh_lo = (npx as f32 * 0.01) as u32;
    let thresh_hi = (npx as f32 * 0.99) as u32;

    let mut cumsum = 0u32;
    let mut bin_lo = 0usize;
    for (b, &count) in hist.iter().enumerate() {
        cumsum += count;
        if cumsum >= thresh_lo {
            bin_lo = b;
            break;
        }
    }

    cumsum = 0;
    let mut bin_hi = 255usize;
    for (b, &count) in hist.iter().enumerate() {
        cumsum += count;
        if cumsum >= thresh_hi {
            bin_hi = b;
            break;
        }
    }

    let p1 = bin_lo as f32 / 255.0_f32;
    let p99 = bin_hi as f32 / 255.0_f32;

    let range = p99 - p1;
    if range < 1e-6 {
        // Flat mid-grey buffer for featureless image
        return Gray8Image::from_raw(width, height, vec![127_u8; npx])
            .expect("buffer length equals width * height");
    }

    let inv_range = 255.0 / range;
    let stretched: Vec<u8> = gamma
        .par_iter()
        .map(|&src| ((src - p1) * inv_range).clamp(0.0, 255.0) as u8)
        .collect();
    Gray8Image::from_raw(width, height, stretched).expect("buffer length equals width * height")
}

/// Hamming distance between two equal-length binary descriptors, computed in
/// 64-bit lanes.  M-LDB descriptors are `BitArray<64>` = 64 bytes = 8 `u64`
/// words, so this does 8 XOR+popcount ops instead of 64 byte-wise ones.
#[inline]
fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    let mut dist = 0u32;
    let (words_a, tail_a) = a.as_chunks::<8>();
    let (words_b, tail_b) = b.as_chunks::<8>();
    for (ca, cb) in words_a.iter().zip(words_b.iter()) {
        let xa = u64::from_ne_bytes(*ca);
        let xb = u64::from_ne_bytes(*cb);
        dist += (xa ^ xb).count_ones();
    }
    // Tail bytes (none for 64-byte descriptors, but keeps this length-agnostic).
    for (x, y) in tail_a.iter().zip(tail_b.iter()) {
        dist += (x ^ y).count_ones();
    }
    dist
}

/// Local implementation of feature matching since `akaze` 0.7.0 removed
/// `match_features`.  Brute-force nearest-descriptor search with Lowe's ratio
/// test.
///
/// A query is kept only when its nearest target descriptor is *strictly* closer
/// than `ratio_test` times the second nearest.  The test needs a second
/// candidate: a target set with a single descriptor yields no matches (there is
/// nothing to be distinctive against), and two target descriptors at the same
/// distance are ambiguous and rejected, even at distance 0.
///
/// The query descriptors (`desc0`) are searched in parallel with rayon, and
/// each pairwise distance uses the 64-bit-lane [`hamming_distance`].  rayon
/// collects the indexed iterator in order, so the output is identical and
/// deterministic relative to a serial implementation.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "Hamming distances of 512-bit descriptors are at most 512, exact in f32"
)]
pub fn match_features(desc0: &[Descriptor], desc1: &[Descriptor], ratio_test: f32) -> Vec<Match> {
    if desc0.is_empty() || desc1.is_empty() {
        return Vec::new();
    }

    desc0
        .par_iter()
        .enumerate()
        .filter_map(|(i, d0)| {
            let bytes0: &[u8] = d0.as_ref();
            let mut best_dist = u32::MAX;
            let mut second_best_dist = u32::MAX;
            let mut best_idx = 0usize;

            for (j, d1) in desc1.iter().enumerate() {
                let dist = hamming_distance(bytes0, d1.as_ref());
                if dist < best_dist {
                    second_best_dist = best_dist;
                    best_dist = dist;
                    best_idx = j;
                } else if dist < second_best_dist {
                    second_best_dist = dist;
                }
            }

            // Lowe's ratio test directly on the discrete Hamming distance.  With
            // fewer than two target descriptors there is no second-best
            // (`u32::MAX`), and the test cannot be applied: reject.
            if second_best_dist != u32::MAX
                && (best_dist as f32) < (second_best_dist as f32) * ratio_test
            {
                Some(Match {
                    index_0: i,
                    index_1: best_idx,
                })
            } else {
                None
            }
        })
        .collect()
}

// ── Public feature-extraction helper ─────────────────────────────────────────

/// Extract AKAZE features from `img` and return `(keypoints, descriptors)`.
///
/// The luma plane is converted to perceptual (gamma-encoded, contrast-stretched)
/// space before detection via [`luma_to_gray8`].  See that function's
/// documentation for the full rationale.  Fusion still receives the original
/// linear-light values stored in `img`.
///
/// This is the building block for the "extract reference once" optimisation:
/// callers extract the reference frame's features once before the per-target
/// loop and reuse them for every target via [`KeypointMatcher::match_target`].
///
/// An image with a side shorter than 64 px is below the detector's size gate:
/// the function returns two empty vectors, **not** an error (it cannot fail).
/// Callers must treat "no features" as "no seed" and fall back to the identity
/// or the intensity refinement alone.
#[must_use]
pub fn extract_ref_features(img: &PlanarImage<f32>) -> (Vec<KeyPoint>, Vec<Descriptor>) {
    if img.width < 64 || img.height < 64 {
        return (Vec::new(), Vec::new());
    }
    let gfi = luma_to_gray8(img);

    // Convert to 8-bit dynamic image instead of 32F.
    let dyn_img = DynamicImage::ImageLuma8(gfi);

    // Config struct was removed; instantiate Akaze directly.
    // Default threshold is 0.001; we reduce it to 0.0003 for macro stacks.
    let akaze = Akaze {
        detector_threshold: 0.000_3_f64,
        ..Default::default()
    };

    akaze.extract(&dyn_img)
}

impl KeypointMatcher {
    /// Extract AKAZE features from two planar images and match them.
    ///
    /// Luma planes are converted to perceptual space **in memory** via
    /// [`luma_to_gray8`] (gamma re-encode + 1st–99th percentile stretch),
    /// then fed to AKAZE with a lowered detector threshold. Fusion is
    /// unaffected — only the temporary detection buffer is gamma-encoded.
    ///
    /// # Errors
    /// Never fails today: an image below the 64-px size gate (see
    /// [`extract_ref_features`]) or a featureless one gives an empty match list,
    /// not an error.  The `Result` is kept so a future failure mode does not
    /// change the signature.
    pub fn extract_and_match(
        ref_img: &PlanarImage<f32>,
        tgt_img: &PlanarImage<f32>,
    ) -> Result<MatchResult, StackerError> {
        let (kps0, desc0) = extract_ref_features(ref_img);
        let (kps1, desc1) = extract_ref_features(tgt_img);

        let matches = match_features(&desc0, &desc1, 0.86);

        Ok(MatchResult {
            matches,
            kps0,
            kps1,
        })
    }

    /// Match pre-extracted reference features against a new target image.
    ///
    /// Use this inside an alignment loop where the reference frame is fixed:
    /// extract the reference features once with [`extract_ref_features`] and
    /// call this method for each target frame instead of calling
    /// [`Self::extract_and_match`] which would redundantly re-extract the
    /// reference on every iteration.
    ///
    /// The target luma is converted to perceptual space identically to the
    /// reference (same [`luma_to_gray8`] + `Akaze` configuration), ensuring
    /// descriptor spaces are compatible.
    ///
    /// # Errors
    /// Never fails today (a [`StackerError`] return is reserved): a target below
    /// the 64-px size gate or without features gives an empty match list, so
    /// callers must handle "no matches" themselves.  The `Result` mirrors
    /// [`Self::extract_and_match`].
    pub fn match_target(
        ref_kps: &[KeyPoint],
        ref_desc: &[Descriptor],
        tgt_img: &PlanarImage<f32>,
    ) -> Result<MatchResult, StackerError> {
        let (kps1, desc1) = extract_ref_features(tgt_img);

        let matches = match_features(ref_desc, &desc1, 0.86);

        // kps0 is a clone of the reference keypoints so callers can resolve
        // match coordinates without holding a borrow on the original slice.
        Ok(MatchResult {
            matches,
            kps0: ref_kps.to_vec(),
            kps1,
        })
    }
}
