use akaze::Akaze;
// Re-exported publicly because `KeyPoint` appears in this module's public API
// (e.g. `extract_ref_features` returns `Vec<KeyPoint>`), so callers must be able
// to name it via `stacker_align::akaze_match::KeyPoint`.
pub use akaze::KeyPoint;
use bitarray::BitArray;
use image::{ImageBuffer, Luma};
use rayon::prelude::*;
use stacker_core::{error::StackerError, image::PlanarImage};

pub struct KeypointMatcher;

/// Match structure, defined locally because `akaze` 0.7.0 removed its built-in `Match` type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Match {
    pub index_0: usize,
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
    pub matches: Vec<Match>,
    pub kps0: Vec<KeyPoint>,
    pub kps1: Vec<KeyPoint>,
}

// ── internal helpers ────────────────────────────────────────────────────────

/// Build an 8-bit `Gray8Image` from the luma plane of a `PlanarImage<f32>`,
/// applying a robust contrast stretch for the AKAZE detector.
///
/// ## Colour-space contract
///
/// The fusion pipeline stores `luma`/`chroma_a`/`chroma_b` in **gamma-encoded
/// (sRGB) light** — no transfer function is applied on load (see
/// `stacker_pipeline::load::dynamic_to_planar` / the GUI's
/// `image_utils::load_as_planar`, both documented as such). Since the luma
/// plane already has perceptual (gamma) contrast, no additional transfer
/// function is needed before feeding it to AKAZE — only the robust
/// contrast stretch below.
///
/// ## Robust 1st–99th percentile contrast stretch
///
/// A specific frame may only use a narrow sub-range of `[0, 1]`. We sort a
/// histogram of 256 bins to find the pixel values at the 1st and 99th
/// percentile, then linearly remap `[p1, p99]` → `[0, 255]`. Using the
/// 1st/99th percentiles makes the stretch robust against isolated specular
/// highlights or stuck pixels.
///
/// ## What is NOT changed
///
/// `PlanarImage::luma`, `chroma_a`, and `chroma_b` are **not** modified.
/// The returned 8-bit image is a temporary, detector-only buffer.
/// Fusion continues to see the original gamma-encoded f32 values.
pub fn luma_to_gray8(img: &PlanarImage<f32>) -> Gray8Image {
    let npx = img.width * img.height;
    let mut gfi = Gray8Image::new(img.width as u32, img.height as u32);

    // ── Clamp to [0, 1]; the plane is already gamma-encoded ─────────────────
    let gamma: Vec<f32> = img.luma.iter().map(|&v| v.clamp(0.0, 1.0)).collect();

    // ── 1st–99th percentile contrast stretch ─────────────────────────────────
    let mut hist = [0u32; 256];
    for &v in &gamma {
        let bin = (v.clamp(0.0, 1.0) * 255.0) as usize;
        hist[bin.min(255)] += 1;
    }

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
        for dst in gfi.pixels_mut() {
            dst.0[0] = 127; // Flat mid-grey buffer for featureless image
        }
        return gfi;
    }

    let inv_range = 255.0 / range;
    for (dst, &src) in gfi.pixels_mut().zip(gamma.iter()) {
        dst.0[0] = ((src - p1) * inv_range).clamp(0.0, 255.0) as u8;
    }

    gfi
}

/// Hamming distance between two equal-length binary descriptors, computed in
/// 64-bit lanes.  M-LDB descriptors are `BitArray<64>` = 64 bytes = 8 `u64`
/// words, so this does 8 XOR+popcount ops instead of 64 byte-wise ones.
#[inline]
fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    let mut dist = 0u32;
    let mut wa = a.chunks_exact(8);
    let mut wb = b.chunks_exact(8);
    for (ca, cb) in wa.by_ref().zip(wb.by_ref()) {
        // `try_into` on an 8-byte `chunks_exact` slice never fails.
        let xa = u64::from_ne_bytes(ca.try_into().unwrap());
        let xb = u64::from_ne_bytes(cb.try_into().unwrap());
        dist += (xa ^ xb).count_ones();
    }
    // Tail bytes (none for 64-byte descriptors, but keeps this length-agnostic).
    for (x, y) in wa.remainder().iter().zip(wb.remainder().iter()) {
        dist += (x ^ y).count_ones();
    }
    dist
}

/// Local implementation of feature matching since `akaze` 0.7.0 removed
/// `match_features`.  Brute-force nearest-descriptor search with Lowe's ratio
/// test.
///
/// The query descriptors (`desc0`) are searched in parallel with rayon, and
/// each pairwise distance uses the 64-bit-lane [`hamming_distance`].  rayon
/// collects the indexed iterator in order, so the output is identical and
/// deterministic relative to a serial implementation.
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

            // Lowe's ratio test directly on the discrete Hamming distance.
            if (best_dist as f32) <= (second_best_dist as f32) * ratio_test {
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
/// The luma plane (already gamma-encoded) is contrast-stretched before
/// detection via [`luma_to_gray8`].  See that function's documentation for
/// the full rationale.  Fusion still receives the original gamma-encoded
/// values stored in `img`.
///
/// This is the building block for the "extract reference once" optimisation:
/// callers extract the reference frame's features once before the per-target
/// loop and reuse them for every target via [`KeypointMatcher::match_target`].
///
/// # Panics
/// Panics if the raw buffer length does not exactly match `width * height`, which is mathematically impossible here since it was extracted from an `ImageBuffer` of those exact dimensions.
pub fn extract_ref_features(img: &PlanarImage<f32>) -> (Vec<KeyPoint>, Vec<Descriptor>) {
    if img.width < 64 || img.height < 64 {
        return (Vec::new(), Vec::new());
    }
    let gfi = luma_to_gray8(img);

    // Convert to 8-bit dynamic image using image023 for akaze
    let width = gfi.width();
    let height = gfi.height();
    let raw = gfi.into_raw();
    let gfi023 =
        image023::ImageBuffer::<image023::Luma<u8>, Vec<u8>>::from_raw(width, height, raw).unwrap();
    let dyn_img = image023::DynamicImage::ImageLuma8(gfi023);

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
    /// Luma planes (already gamma-encoded) are contrast-stretched **in
    /// memory** via [`luma_to_gray8`] (1st–99th percentile stretch), then
    /// fed to AKAZE with a lowered detector threshold. Fusion is
    /// unaffected — only the temporary detection buffer is used.
    ///
    /// # Errors
    /// Returns [`StackerError::AlignmentFailed`] when the image is too small
    /// for AKAZE to allocate any evolution steps.
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
    /// The target luma is contrast-stretched identically to the reference
    /// (same [`luma_to_gray8`] + `Akaze` configuration), ensuring descriptor
    /// spaces are compatible.
    ///
    /// # Errors
    /// Returns [`StackerError::AlignmentFailed`] — currently unused (AKAZE
    /// returns an empty match list for trivial inputs rather than an error),
    /// but kept for API symmetry with [`Self::extract_and_match`].
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
