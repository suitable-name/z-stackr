// Test-only helpers use short mathematical names (w/h/l/a/b) and
// usize->f32 pixel-coordinate casts throughout — pedantic/nursery noise with
// no behavioural relevance to what these tests assert; see the workspace
// lint policy note in other test files for the same rationale.
#![allow(clippy::many_single_char_names, clippy::cast_precision_loss)]

use stacker_algo::{
    apex::pyramid::apply_gaussian_blur,
    strata::{StrataParams, fuse_strata, saliency::compute_argmax},
};
use stacker_core::image::PlanarImage;

fn make_const_image(w: usize, h: usize, l: f32, a: f32, b: f32) -> PlanarImage<f32> {
    PlanarImage {
        width: w,
        height: h,
        luma: vec![l; w * h],
        chroma_a: vec![a; w * h],
        chroma_b: vec![b; w * h],
    }
}

fn make_textured_image(w: usize, h: usize, phase: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let v = ((x as f32).mul_add(0.3, phase).sin()
                * phase.mul_add(1.1, y as f32 * 0.25).cos())
            .mul_add(0.5, 0.5)
            .clamp(0.0, 1.0);
            img.luma[i] = v;
            img.chroma_a[i] = (v - 0.5) * 0.3;
            img.chroma_b[i] = (0.5 - v) * 0.2;
        }
    }
    img
}

fn half_sharp_image(w: usize, h: usize, sharp_on_left: bool) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let in_sharp_region = if sharp_on_left { x < w / 2 } else { x >= w / 2 };
            img.luma[i] = if in_sharp_region {
                if (x + y) % 2 == 0 { 0.9 } else { 0.1 }
            } else {
                0.5
            };
        }
    }
    img
}

fn mean_abs_laplacian(img: &PlanarImage<f32>, x0: usize, x1: usize, height: usize) -> f32 {
    let width = img.width;
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for y in 0..height {
        for x in x0..x1 {
            let up = img.luma[y.saturating_sub(1) * width + x];
            let down = img.luma[(y + 1).min(height - 1) * width + x];
            let left = img.luma[y * width + x.saturating_sub(1)];
            let right = img.luma[y * width + (x + 1).min(width - 1)];
            let center = img.luma[y * width + x];
            sum += 4.0f32.mul_add(-center, up + down + left + right).abs();
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

#[test]
fn strata_constant_stack_is_identity() {
    let (w, h) = (24, 24);
    let (l, a, b) = (0.42_f32, 0.1_f32, -0.05_f32);
    let img = make_const_image(w, h, l, a, b);
    let frames: Vec<_> = (0..4).map(|_| img.clone()).collect();
    let params = StrataParams {
        base_radius: 8,
        ..Default::default()
    };
    let fused = fuse_strata(&frames, &params);

    for &v in &fused.luma {
        assert!((v - l).abs() < 1e-5, "luma: got {v}, want {l}");
    }
    for &v in &fused.chroma_a {
        assert!((v - a).abs() < 1e-5, "chroma_a: got {v}, want {a}");
    }
    for &v in &fused.chroma_b {
        assert!((v - b).abs() < 1e-5, "chroma_b: got {v}, want {b}");
    }
}

#[test]
fn strata_identical_frames_roundtrip() {
    let (w, h) = (32, 32);
    let img = make_textured_image(w, h, 0.7);
    let frames: Vec<_> = (0..5).map(|_| img.clone()).collect();
    let params = StrataParams {
        base_radius: 6,
        ..Default::default()
    };
    let fused = fuse_strata(&frames, &params);

    let max_diff = fused
        .luma
        .iter()
        .zip(img.luma.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "identical-frames max abs diff {max_diff} >= 1e-3"
    );
}

#[test]
fn strata_two_half_sharp_frames() {
    let (w, h) = (40, 40);
    let frame_a = half_sharp_image(w, h, true);
    let frame_b = half_sharp_image(w, h, false);
    let params = StrataParams {
        base_radius: 6,
        ..Default::default()
    };
    let fused = fuse_strata(&[frame_a.clone(), frame_b.clone()], &params);

    let left_fused = mean_abs_laplacian(&fused, 0, w / 2, h);
    let left_b = mean_abs_laplacian(&frame_b, 0, w / 2, h);
    assert!(
        left_fused > left_b,
        "left half: fused ({left_fused}) should be sharper than the flat frame B ({left_b})"
    );

    let right_fused = mean_abs_laplacian(&fused, w / 2, w, h);
    let right_a = mean_abs_laplacian(&frame_a, w / 2, w, h);
    assert!(
        right_fused > right_a,
        "right half: fused ({right_fused}) should be sharper than the flat frame A ({right_a})"
    );
}

#[test]
fn strata_weights_pick_lowest_index_on_ties() {
    let (w, h) = (20, 20);
    let img = make_textured_image(w, h, 1.3);
    let params = StrataParams {
        base_radius: 5,
        ..Default::default()
    };
    let single = fuse_strata(std::slice::from_ref(&img), &params);
    let two = fuse_strata(&[img.clone(), img.clone()], &params);

    let diff = single
        .luma
        .iter()
        .zip(two.luma.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        diff < 1e-3,
        "two identical frames should fuse like one: diff {diff}"
    );

    let (argmax_idx, _argmax_val) = compute_argmax(&[img.clone(), img], w, h);
    assert!(
        argmax_idx.iter().all(|&idx| idx == 0),
        "exact ties must resolve to the lowest frame index"
    );
}

#[test]
fn strata_flat_region_fallback_is_uniform() {
    let (w, h) = (16, 16);
    let val = 0.33_f32;
    let frames: Vec<_> = (0..3)
        .map(|_| make_const_image(w, h, val, 0.0, 0.0))
        .collect();
    let params = StrataParams {
        base_radius: 4,
        ..Default::default()
    };
    let fused = fuse_strata(&frames, &params);

    for &v in &fused.luma {
        assert!(v.is_finite(), "flat stack must never produce NaN/inf");
        assert!(
            (v - val).abs() < 1e-4,
            "flat stack must fuse to its own value, got {v}"
        );
    }
}

// ── Deep-stack detail-dilution regression (Defect 1) ─────────────────────
//
// Regression coverage for the "washed-out, watercolour" deep-stack defect:
// with N ~ dozens of frames, the guided filter's edge-aware feathering
// leaks a small positive `w_detail` to every near-winner frame, not just
// the true local winner. Those O(0.01-0.05) leaks sum to O(1) across ~49
// losers, diluting the winner's sharp detail layer with many defocused
// ones. The fix (`DETAIL_WEIGHT_EXPONENT` = 3, cubing `w_detail` after its
// `[0, 1]` clamp) re-concentrates the competition. This test builds a
// synthetic N=12 deep stack -- one high-frequency ground-truth texture,
// each frame sharp in its own horizontal band and heavily blurred
// elsewhere -- and checks the fused result keeps most of the ground
// truth's detail energy, and clearly beats a plain N-frame average (the
// "watercolour" baseline the unfixed math degenerates toward).

const DEEP_STACK_FRAMES: usize = 12;

/// High-frequency ground-truth texture: a product of two sinusoids at
/// different, fairly high spatial frequencies, so every band has strong,
/// genuine high-frequency detail for the fix to preserve or lose.
fn make_high_freq_ground_truth(w: usize, h: usize) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let v = ((x as f32) * 0.9).sin() * ((y as f32) * 0.8).cos();
            img.luma[i] = v.mul_add(0.5, 0.5).clamp(0.0, 1.0);
        }
    }
    img
}

/// Heavily blur `img` by repeated application of the existing 5-tap
/// separable blur (`apex::pyramid::apply_gaussian_blur`) -- the same
/// primitive Strata's own saliency pass reuses (see `strata::mod`'s
/// `BLUR_PASSES` doc comment), just applied many more times here to
/// produce a strongly defocused reference "out of band" image.
fn heavily_blur(img: &PlanarImage<f32>, passes: usize) -> PlanarImage<f32> {
    let mut out = PlanarImage {
        width: img.width,
        height: img.height,
        luma: img.luma.clone(),
        chroma_a: img.chroma_a.clone(),
        chroma_b: img.chroma_b.clone(),
    };
    for _ in 0..passes {
        out = apply_gaussian_blur(&out);
    }
    out
}

/// Build one deep-stack frame: sharp (equal to `ground_truth`) within its
/// own horizontal band `[band_idx * h/n .. (band_idx+1) * h/n)`, and the
/// heavily-blurred version everywhere else -- modelling a focus stack
/// where each frame is only in-focus over a slice of the scene.
fn make_band_sharp_frame(
    ground_truth: &PlanarImage<f32>,
    blurred: &PlanarImage<f32>,
    band_idx: usize,
    n_bands: usize,
) -> PlanarImage<f32> {
    let (w, h) = (ground_truth.width, ground_truth.height);
    let band_h = h / n_bands;
    let y0 = band_idx * band_h;
    let y1 = if band_idx + 1 == n_bands {
        h
    } else {
        (band_idx + 1) * band_h
    };

    let mut out = blurred.clone();
    for y in y0..y1 {
        for x in 0..w {
            let i = y * w + x;
            out.luma[i] = ground_truth.luma[i];
        }
    }
    out
}

/// Mean |4-neighbour Laplacian| over `y0..y1` (all columns), the same
/// focus-energy metric `mean_abs_laplacian` uses elsewhere in this file,
/// but restricted to a row range so band-boundary rows (where a sharp band
/// meets a heavily-blurred neighbour, a discontinuity that is an artefact
/// of this synthetic construction rather than the fusion algorithm) can be
/// excluded from the measurement.
fn mean_abs_laplacian_rows(img: &PlanarImage<f32>, y0: usize, y1: usize) -> f32 {
    let width = img.width;
    let height = img.height;
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for y in y0..y1 {
        for x in 0..width {
            let up = img.luma[y.saturating_sub(1) * width + x];
            let down = img.luma[(y + 1).min(height - 1) * width + x];
            let left = img.luma[y * width + x.saturating_sub(1)];
            let right = img.luma[y * width + (x + 1).min(width - 1)];
            let center = img.luma[y * width + x];
            sum += 4.0f32.mul_add(-center, up + down + left + right).abs();
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

/// Plain (unweighted) N-frame average -- the "watercolour" baseline: no
/// per-frame weighting at all, just `mean_i(frame_i)`. This is what the
/// unfixed detail-weight math effectively degenerates toward on deep
/// stacks (dozens of near-equal small leaked weights average away nearly
/// all real detail), so it is the right thing for the fixed Strata output
/// to clearly beat, not just "do better than one input frame".
fn plain_average(frames: &[PlanarImage<f32>]) -> PlanarImage<f32> {
    let (w, h) = (frames[0].width, frames[0].height);
    let len = w * h;
    let n = frames.len() as f32;
    let mut out = PlanarImage::new(w, h);
    for frame in frames {
        for i in 0..len {
            out.luma[i] += frame.luma[i] / n;
        }
    }
    out
}

/// Mean interior detail energy (see `strata_deep_stack_preserves_detail_energy`'s
/// doc comment for the band/margin measurement rationale) of a Strata fuse
/// of `frames` at a given `detail_focus` (`StrataParams::detail_focus`).
/// Shared by the base regression test and the monotonicity test below so
/// both measure identically.
fn deep_stack_fused_interior_energy(
    frames: &[PlanarImage<f32>],
    _w: usize,
    h: usize,
    detail_focus: u32,
) -> f32 {
    let params = StrataParams {
        base_radius: 8,
        detail_focus,
    };
    let fused = fuse_strata(frames, &params);

    let band_h = h / DEEP_STACK_FRAMES;
    let margin = (band_h / 4).max(1);
    let mut fused_energy = 0.0_f32;
    let mut n_bands_measured = 0usize;
    for band_idx in 0..DEEP_STACK_FRAMES {
        let y0 = band_idx * band_h + margin;
        let y1 = ((band_idx + 1) * band_h).saturating_sub(margin).max(y0 + 1);
        if y1 <= y0 || y1 > h {
            continue;
        }
        fused_energy += mean_abs_laplacian_rows(&fused, y0, y1);
        n_bands_measured += 1;
    }
    assert!(n_bands_measured > 0, "no interior band rows measured");
    fused_energy / n_bands_measured as f32
}

#[test]
fn strata_deep_stack_preserves_detail_energy() {
    let (w, h) = (48, 48);
    let ground_truth = make_high_freq_ground_truth(w, h);
    // Heavy blur: enough passes that the out-of-band regions are
    // genuinely defocused (comparable to a real out-of-focus frame), not
    // just softened.
    let blurred = heavily_blur(&ground_truth, 12);

    let frames: Vec<_> = (0..DEEP_STACK_FRAMES)
        .map(|band_idx| make_band_sharp_frame(&ground_truth, &blurred, band_idx, DEEP_STACK_FRAMES))
        .collect();

    let params = StrataParams {
        base_radius: 8,
        ..Default::default()
    };
    let fused = fuse_strata(&frames, &params);
    let baseline = plain_average(&frames);

    // Measure detail energy over the interior of each band, away from
    // band-boundary rows (excluded via a margin), so the comparison is
    // about how well genuine in-band detail survives fusion -- not an
    // artefact of the synthetic hard band edges.
    let band_h = h / DEEP_STACK_FRAMES;
    let margin = (band_h / 4).max(1);
    let mut gt_energy = 0.0_f32;
    let mut fused_energy = 0.0_f32;
    let mut baseline_energy = 0.0_f32;
    let mut n_bands_measured = 0usize;
    for band_idx in 0..DEEP_STACK_FRAMES {
        let y0 = band_idx * band_h + margin;
        let y1 = ((band_idx + 1) * band_h).saturating_sub(margin).max(y0 + 1);
        if y1 <= y0 || y1 > h {
            continue;
        }
        gt_energy += mean_abs_laplacian_rows(&ground_truth, y0, y1);
        fused_energy += mean_abs_laplacian_rows(&fused, y0, y1);
        baseline_energy += mean_abs_laplacian_rows(&baseline, y0, y1);
        n_bands_measured += 1;
    }
    assert!(n_bands_measured > 0, "no interior band rows measured");
    gt_energy /= n_bands_measured as f32;
    fused_energy /= n_bands_measured as f32;
    baseline_energy /= n_bands_measured as f32;

    // Thresholds tuned to what the fixed implementation (gamma = 3, the
    // `detail_focus` default) actually achieves on this synthetic stack,
    // with margin against the unfixed (gamma = 1, no-op) math measured
    // directly by temporarily hard-coding `raise_detail_weight` to return
    // `w` unchanged:
    //   fixed:   fused/gt = 0.502, fused/baseline = 6.02
    //   unfixed: fused/gt = 0.291, fused/baseline = 3.49
    // 0.4x and 5.0x sit strictly between the fixed and unfixed ratios, so
    // this test passes with the fix and fails without it (verified by
    // temporarily reverting `raise_detail_weight` to a `gamma = 1` no-op:
    // ratio_gt 0.291 < 0.4, ratio_baseline 3.49 < 5.0 - both assertions
    // below fail on the unfixed math).
    assert!(
        fused_energy >= 0.4 * gt_energy,
        "fused interior detail energy ({fused_energy}) should retain at least \
         0.4x the ground truth's ({gt_energy}), got ratio {}",
        fused_energy / gt_energy
    );
    assert!(
        fused_energy >= 5.0 * baseline_energy,
        "fused interior detail energy ({fused_energy}) should clearly beat \
         (>= 5x) the plain N-frame average baseline ({baseline_energy}), got \
         ratio {}",
        fused_energy / baseline_energy.max(1e-9)
    );
}

/// Regression coverage for the "Detail focus" setting
/// (`StrataParams::detail_focus`, exposed as `strata_detail_focus`):
/// raising the detail-weight exponent must only ever concentrate detail
/// energy more, never less, on the same deep stack — `detail_focus = 5`
/// should retain at least as much interior detail energy as `3` (the
/// default), which in turn should retain at least as much as `1` (the
/// original, no-op Li-Kang-Hu behaviour). This directly exercises the
/// "higher = crisper depth edges / more detail retention on deep stacks"
/// contract the setting's docs/tooltips promise.
#[test]
fn strata_deep_stack_detail_focus_is_monotonic() {
    let (w, h) = (48, 48);
    let ground_truth = make_high_freq_ground_truth(w, h);
    let blurred = heavily_blur(&ground_truth, 12);
    let frames: Vec<_> = (0..DEEP_STACK_FRAMES)
        .map(|band_idx| make_band_sharp_frame(&ground_truth, &blurred, band_idx, DEEP_STACK_FRAMES))
        .collect();

    let energy_1 = deep_stack_fused_interior_energy(&frames, w, h, 1);
    let energy_3 = deep_stack_fused_interior_energy(&frames, w, h, 3);
    let energy_5 = deep_stack_fused_interior_energy(&frames, w, h, 5);

    assert!(
        energy_5 >= energy_3,
        "detail_focus=5 interior detail energy ({energy_5}) should be >= \
         detail_focus=3's ({energy_3})"
    );
    assert!(
        energy_3 >= energy_1,
        "detail_focus=3 interior detail energy ({energy_3}) should be >= \
         detail_focus=1's ({energy_1})"
    );
}
