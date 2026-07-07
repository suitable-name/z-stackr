#![cfg(feature = "akaze")]
#![allow(
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::imprecise_flops
)]

use image::{ColorType, DynamicImage, GenericImageView};
#[cfg(feature = "akaze")]
use stacker_align::{
    akaze_match::{KeypointMatcher, extract_ref_features},
    ransac::{AlignmentEstimator, AlignmentMode},
};
use stacker_align::{
    refine::{BoundedRefineOptions, refine_alignment_registration},
    transform::warp_image_clamped,
};
use stacker_core::{color::srgb_to_linear, image::PlanarImage};
use std::path::{Path, PathBuf};

/// Long-side cap (px) applied to every sample frame before processing.
///
/// `test_align_sample_images` is a *smoke* test — it asserts nothing about
/// alignment accuracy, only that the AKAZE → RANSAC → refine → warp pipeline
/// runs end-to-end and produces output files. AKAZE feature extraction on
/// full-resolution macro frames is the dominant cost (seconds per frame), so we
/// shrink each frame to keep the whole test in the sub-second range.
const MAX_TEST_DIM: u32 = 256;

/// Number of target frames (after the reference) the smoke test processes.
const MAX_TEST_TARGETS: usize = 2;

fn dynamic_to_planar(img: &DynamicImage) -> PlanarImage<f32> {
    let (w, h) = img.dimensions();
    let mut planar = PlanarImage::new(w as usize, h as usize);

    let is_16bit = matches!(
        img.color(),
        ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16
    );

    if is_16bit {
        let rgb16 = img.to_rgb16();
        for (i, p) in rgb16.pixels().enumerate() {
            let r = srgb_to_linear(f32::from(p[0]) / 65_535.0);
            let g = srgb_to_linear(f32::from(p[1]) / 65_535.0);
            let b = srgb_to_linear(f32::from(p[2]) / 65_535.0);
            planar.luma[i] = 0.299 * r + 0.587 * g + 0.114 * b;
            planar.chroma_a[i] = -0.168_74 * r - 0.331_26 * g + 0.5 * b;
            planar.chroma_b[i] = 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        }
    } else {
        let rgb8 = img.to_rgb8();
        for (i, p) in rgb8.pixels().enumerate() {
            let r = srgb_to_linear(f32::from(p[0]) / 255.0);
            let g = srgb_to_linear(f32::from(p[1]) / 255.0);
            let b = srgb_to_linear(f32::from(p[2]) / 255.0);
            planar.luma[i] = 0.299 * r + 0.587 * g + 0.114 * b;
            planar.chroma_a[i] = -0.168_74 * r - 0.331_26 * g + 0.5 * b;
            planar.chroma_b[i] = 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        }
    }

    planar
}

/// Downscale a frame so its long side is at most `max_dim` px. Frames
/// already small enough are returned unchanged.
fn downscale_to(img: DynamicImage, max_dim: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    let long = w.max(h);
    if long <= max_dim {
        return img;
    }
    let scale = f64::from(max_dim) / f64::from(long);
    let nw = (f64::from(w) * scale).round() as u32;
    let nh = (f64::from(h) * scale).round() as u32;
    img.resize(nw, nh, image::imageops::FilterType::Triangle)
}

/// Downscale a frame so its long side is at most [`MAX_TEST_DIM`] px. Frames
/// already small enough are returned unchanged.
fn downscale_for_test(img: DynamicImage) -> DynamicImage {
    downscale_to(img, MAX_TEST_DIM)
}

fn collect_image_paths(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("frame_")
                && p.extension().is_some_and(|e| e == "jpg")
        })
        .collect();
    paths.sort();
    paths
}

#[test]
#[cfg(feature = "akaze")]
fn test_align_sample_images() {
    let sample_dir = Path::new("../../sample");
    let paths = collect_image_paths(sample_dir);
    if paths.is_empty() {
        return;
    }

    let aligned_dir = std::env::temp_dir()
        .join("focus-stacker-rs")
        .join("sample_aligned");
    std::fs::create_dir_all(&aligned_dir).unwrap();

    let ref_dyn = downscale_for_test(image::open(&paths[0]).unwrap());
    let ref_frame = dynamic_to_planar(&ref_dyn);

    println!(
        "{:?}",
        aligned_dir.join(format!("aligned_frame_{:04}.jpg", 0))
    );

    ref_dyn
        .save(aligned_dir.join(format!("aligned_frame_{:04}.jpg", 0)))
        .unwrap();

    let (ref_kps, ref_desc) = extract_ref_features(&ref_frame);

    for (i, path) in paths.iter().enumerate().skip(1).take(MAX_TEST_TARGETS) {
        let dyn_img = downscale_for_test(image::open(path).unwrap());
        let frame = dynamic_to_planar(&dyn_img);

        let mr = KeypointMatcher::match_target(&ref_kps, &ref_desc, &frame).unwrap();

        let coarse_matrix = AlignmentEstimator::compute_matrix(
            &mr.matches[..],
            &mr.kps0,
            &mr.kps1,
            AlignmentMode::TranslationAndScale,
        )
        .unwrap_or_else(|_| stacker_align::Matrix3::identity());

        let final_matrix = refine_alignment_registration(
            &ref_frame,
            &frame,
            &coarse_matrix,
            &BoundedRefineOptions {
                max_iterations: 100,
                ..BoundedRefineOptions::default()
            },
        )
        .unwrap_or(coarse_matrix);

        let warped_planar = warp_image_clamped(&frame, &final_matrix).unwrap_or(frame);

        let w = warped_planar.width as u32;
        let h = warped_planar.height as u32;
        let mut out_img = image::RgbImage::new(w, h);
        for (i_px, p) in out_img.pixels_mut().enumerate() {
            let luma = warped_planar.luma[i_px];
            let cb = warped_planar.chroma_a[i_px];
            let cr = warped_planar.chroma_b[i_px];

            let r = (luma + 1.402 * cr).clamp(0.0, 1.0);
            let g = (luma - 0.344_136 * cb - 0.714_136 * cr).clamp(0.0, 1.0);
            let b = (luma + 1.772 * cb).clamp(0.0, 1.0);

            let r_srgb = if r <= 0.003_130_8 {
                r * 12.92
            } else {
                1.055 * r.powf(1.0 / 2.4) - 0.055
            };
            let g_srgb = if g <= 0.003_130_8 {
                g * 12.92
            } else {
                1.055 * g.powf(1.0 / 2.4) - 0.055
            };
            let b_srgb = if b <= 0.003_130_8 {
                b * 12.92
            } else {
                1.055 * b.powf(1.0 / 2.4) - 0.055
            };

            p[0] = (r_srgb * 255.0) as u8;
            p[1] = (g_srgb * 255.0) as u8;
            p[2] = (b_srgb * 255.0) as u8;
        }

        out_img
            .save(aligned_dir.join(format!("aligned_frame_{i:04}.jpg")))
            .unwrap();
    }
}

// ── Subject stability regression test ───────────────────────────────────────
//
// Regression coverage for a previously-diagnosed bug: the subject (e.g. a
// gemstone) visibly jumped frame to frame when scrubbing aligned output,
// traced to an overly-aggressive finest-level Nelder-Mead iteration cap. The
// fix lives in the shared `stacker_align::pipeline::align_frame` dispatch
// (coarse-to-fine iteration schedule + bounded registration); this test
// exercises that exact dispatch — the same one both the CLI and GUI call —
// end-to-end against the full real sample stack, so a future regression in
// either the iteration schedule or the dispatch itself is caught here.

/// Zero-mean normalized cross-correlation between a fixed template patch and
/// an equally sized window of `image` anchored at `(x, y)` (the window's
/// top-left corner).
fn ncc_at(
    image: &[f32],
    img_w: usize,
    x: usize,
    y: usize,
    patch: &[f32],
    patch_w: usize,
    patch_h: usize,
) -> f32 {
    let n = (patch_w * patch_h) as f32;
    let mut sum_t = 0.0_f32;
    let mut sum_i = 0.0_f32;
    for py in 0..patch_h {
        for px in 0..patch_w {
            sum_t += patch[py * patch_w + px];
            sum_i += image[(y + py) * img_w + (x + px)];
        }
    }
    let mean_t = sum_t / n;
    let mean_i = sum_i / n;

    let mut num = 0.0_f32;
    let mut den_t = 0.0_f32;
    let mut den_i = 0.0_f32;
    for py in 0..patch_h {
        for px in 0..patch_w {
            let dt = patch[py * patch_w + px] - mean_t;
            let di = image[(y + py) * img_w + (x + px)] - mean_i;
            num += dt * di;
            den_t += dt * dt;
            den_i += di * di;
        }
    }
    if den_t <= 1.0e-8 || den_i <= 1.0e-8 {
        return -1.0;
    }
    num / (den_t.sqrt() * den_i.sqrt())
}

/// Locate the best-matching centre position of `patch` within `image` via an
/// integer-pixel NCC search over a `± search_radius` window around the
/// patch's expected centre `(cx, cy)`.
#[allow(clippy::too_many_arguments)]
fn locate_patch(
    image: &[f32],
    img_w: usize,
    img_h: usize,
    cx: usize,
    cy: usize,
    search_radius: i32,
    patch: &[f32],
    patch_w: usize,
    patch_h: usize,
) -> (f32, f32) {
    let _ = img_h;
    let half_w = (patch_w / 2) as i32;
    let half_h = (patch_h / 2) as i32;
    let mut best_score = f32::NEG_INFINITY;
    let mut best = (cx as f32, cy as f32);
    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let tx = cx as i32 + dx - half_w;
            let ty = cy as i32 + dy - half_h;
            if tx < 0 || ty < 0 {
                continue;
            }
            let (tx, ty) = (tx as usize, ty as usize);
            if tx + patch_w > img_w || ty + patch_h > img_h {
                continue;
            }
            let score = ncc_at(image, img_w, tx, ty, patch, patch_w, patch_h);
            if score > best_score {
                best_score = score;
                best = ((cx as i32 + dx) as f32, (cy as i32 + dy) as f32);
            }
        }
    }
    best
}

/// Simple local sharpness proxy (Laplacian energy, box-smoothed) used only to
/// locate a strongly in-focus "subject" region in the reference frame — good
/// enough for this test without pulling in the full `stacker-algo` SML
/// implementation as a dev-dependency.
fn local_sharpness_map(luma: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut lap = vec![0.0_f32; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            let c = luma[i];
            let dxx = luma[i - 1] + luma[i + 1] - 2.0 * c;
            let dyy = luma[i - w] + luma[i + w] - 2.0 * c;
            lap[i] = dxx * dxx + dyy * dyy;
        }
    }
    let radius = 5usize;
    let mut smoothed = vec![0.0_f32; w * h];
    for y in 0..h {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius).min(h - 1);
        for x in 0..w {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius).min(w - 1);
            let mut sum = 0.0_f32;
            let mut count = 0usize;
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    sum += lap[yy * w + xx];
                    count += 1;
                }
            }
            smoothed[y * w + x] = sum / count as f32;
        }
    }
    smoothed
}

/// Locate the sharpest point within the central two-thirds of the frame (the
/// margin avoids picking up vignette/border artefacts as the "subject").
fn find_sharpest_location(luma: &[f32], w: usize, h: usize) -> (usize, usize) {
    let energy = local_sharpness_map(luma, w, h);
    let margin_x = w / 6;
    let margin_y = h / 6;
    let mut best = (w / 2, h / 2);
    let mut best_val = f32::NEG_INFINITY;
    for y in margin_y..h - margin_y {
        for x in margin_x..w - margin_x {
            let v = energy[y * w + x];
            if v > best_val {
                best_val = v;
                best = (x, y);
            }
        }
    }
    best
}

/// After aligning every frame in the full sample stack to the reference
/// frame with the same shared [`stacker_align::align_frame`] dispatch the
/// CLI and GUI use, a fixed subject feature (located via local sharpness in
/// the reference) must land at very nearly the same pixel position in every
/// aligned frame — both in absolute terms and frame-to-frame.
#[test]
#[cfg(feature = "akaze")]
fn test_alignment_subject_stability_across_sample_stack() {
    use rayon::prelude::*;
    use stacker_align::pipeline::akaze_mode_for_alignment;
    use stacker_core::settings::AlignmentModeSetting;

    let sample_dir = Path::new("../../sample");
    let paths = collect_image_paths(sample_dir);
    if paths.is_empty() {
        return;
    }
    assert!(
        paths.len() >= 20,
        "expected the full ~28-frame sample stack, found {} frames",
        paths.len()
    );

    // A smaller working resolution than the plain pipeline smoke test above:
    // this test runs the full bounded-registration dispatch (coarse-to-fine
    // pyramid + coarsest-level random restarts) — much heavier per frame
    // than the smoke test's single-shot `refine_alignment_registration` call
    // — across all ~28 frames, so it needs to stay cheap per frame to keep
    // total runtime reasonable. Tolerances are scaled down accordingly.
    const STABILITY_TEST_DIM: u32 = 160;

    // The primary regression guard is the *frame-to-frame* jump: the
    // originally-reported bug was the subject visibly jumping between
    // consecutive aligned frames, which this catches directly. A separate,
    // much looser absolute-deviation backstop only guards against gross
    // pipeline breakage (e.g. alignment doing nothing at all) — measured
    // against this real 28-frame macro stack, a few pixels of *smooth,
    // monotonic* cumulative drift toward the far end of the stack is normal
    // (natural focus breathing / progressively weaker AKAZE overlap with the
    // reference) and must not fail the test; a sudden multi-pixel jump
    // between two adjacent frames must.
    const MAX_ABS_DEVIATION_PX: f32 = 6.0;
    const MAX_FRAME_TO_FRAME_JUMP_PX: f32 = 1.5;
    const PATCH_SIZE: usize = 15;
    const SEARCH_RADIUS: i32 = 6;

    let ref_dyn = downscale_to(image::open(&paths[0]).unwrap(), STABILITY_TEST_DIM);
    let ref_frame = dynamic_to_planar(&ref_dyn);
    let (ref_kps, ref_desc) = extract_ref_features(&ref_frame);

    let (sub_x, sub_y) = find_sharpest_location(&ref_frame.luma, ref_frame.width, ref_frame.height);
    let half = PATCH_SIZE / 2;
    // The margin used by `find_sharpest_location` already keeps the peak
    // away from the border, so the patch is guaranteed to fit.
    let mut ref_patch = vec![0.0_f32; PATCH_SIZE * PATCH_SIZE];
    for py in 0..PATCH_SIZE {
        for px in 0..PATCH_SIZE {
            let x = sub_x - half + px;
            let y = sub_y - half + py;
            ref_patch[py * PATCH_SIZE + px] = ref_frame.luma[y * ref_frame.width + x];
        }
    }
    let ref_pos = (sub_x as f32, sub_y as f32);

    let akaze_mode = akaze_mode_for_alignment(AlignmentModeSetting::Registration);

    // Each target frame is aligned independently to the same reference, so
    // this is embarrassingly parallel — run it across all available cores to
    // keep the full 28-frame stack fast.
    let mut indexed_positions: Vec<(usize, (f32, f32))> = paths
        .iter()
        .enumerate()
        .skip(1)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(idx, path)| {
            let dyn_img = downscale_to(image::open(path).unwrap(), STABILITY_TEST_DIM);
            let frame = dynamic_to_planar(&dyn_img);

            let mr = KeypointMatcher::match_target(&ref_kps, &ref_desc, &frame).unwrap();
            let coarse_matrix =
                AlignmentEstimator::compute_matrix(&mr.matches[..], &mr.kps0, &mr.kps1, akaze_mode)
                    .unwrap_or_else(|_| stacker_align::Matrix3::identity());

            let (_matrix, warped) = stacker_align::align_frame(
                frame,
                &ref_frame,
                coarse_matrix,
                AlignmentModeSetting::Registration,
                stacker_core::settings::OptimizerSetting::NelderMead,
                true,
                idx,
                None,
            );

            let pos = locate_patch(
                &warped.luma,
                warped.width,
                warped.height,
                sub_x,
                sub_y,
                SEARCH_RADIUS,
                &ref_patch,
                PATCH_SIZE,
                PATCH_SIZE,
            );
            (idx, pos)
        })
        .collect();
    indexed_positions.sort_by_key(|&(idx, _)| idx);

    let mut positions = vec![ref_pos];
    positions.extend(indexed_positions.into_iter().map(|(_, pos)| pos));

    let mut max_abs_dev = 0.0_f32;
    let mut max_jump = 0.0_f32;
    for (i, &(x, y)) in positions.iter().enumerate() {
        let dev = ((x - ref_pos.0).powi(2) + (y - ref_pos.1).powi(2)).sqrt();
        if dev > max_abs_dev {
            max_abs_dev = dev;
        }
        let jump = if i > 0 {
            let (px, py) = positions[i - 1];
            let j = ((x - px).powi(2) + (y - py).powi(2)).sqrt();
            if j > max_jump {
                max_jump = j;
            }
            j
        } else {
            0.0
        };
        eprintln!("frame {i}: pos=({x:.2},{y:.2}) abs_dev={dev:.2}px jump={jump:.2}px");
    }

    assert!(
        max_abs_dev < MAX_ABS_DEVIATION_PX,
        "subject feature drifted {max_abs_dev:.2}px from its reference-frame position \
         across the aligned stack (limit {MAX_ABS_DEVIATION_PX}px) — alignment is not \
         holding the subject stationary"
    );
    assert!(
        max_jump < MAX_FRAME_TO_FRAME_JUMP_PX,
        "subject feature jumped {max_jump:.2}px between two consecutive aligned frames \
         (limit {MAX_FRAME_TO_FRAME_JUMP_PX}px) — this is the 'gemstone jumps frame to \
         frame' regression"
    );
}
