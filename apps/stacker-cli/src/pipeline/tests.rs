#![expect(
    clippy::suboptimal_flops,
    reason = "test fixtures and image metrics: readable formulas, not performance code"
)]

use super::*;
use image::GenericImageView;
use stacker_core::{
    image::PlanarImage,
    memory::{enumerate_tiles, extract_tile, paste_interior},
    settings::{ImageSavingSettings, OutputFormat, StackingSettings},
};
use stacker_pipeline::{
    fuse::{apex as run_apex, focus_maps, relief as run_relief, relief_threshold, relief_tile},
    load::planar_from_dynamic,
    output::encode,
    tiled::ScratchDir,
    tiling::{Geometry, Method, plan_tiles},
};

use crate::args::CliArgs;

/// Build a tiny synthetic `PlanarImage` with a checkerboard-like pattern.
#[expect(
    clippy::cast_precision_loss,
    reason = "pixel coordinates of a tiny synthetic image"
)]
fn make_synthetic(width: usize, height: usize, offset: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let pat = ((x + y) % 2) as f32;
            img.luma[i] = (pat + offset).clamp(0.0, 1.0);
            img.chroma_a[i] = 0.1 * pat;
            img.chroma_b[i] = 0.2 * pat;
        }
    }
    img
}

#[test]
#[expect(
    clippy::cast_precision_loss,
    reason = "frame indices 0..3 are exact in f32"
)]
fn test_apex_pipeline_produces_valid_output() {
    let imgs: Vec<PlanarImage<f32>> = (0..3_i32)
        .map(|i| make_synthetic(16, 16, i as f32 * 0.1))
        .collect();
    let settings = stacker_core::settings::StackingSettings::default();
    let out = run_apex(&imgs, &settings);
    assert_eq!(out.width, 16);
    assert_eq!(out.height, 16);
    assert!(
        out.luma.iter().all(|v| v.is_finite()),
        "luma must be finite"
    );
    assert!(
        out.chroma_a.iter().all(|v| v.is_finite()),
        "chroma_a must be finite"
    );
    assert!(
        out.chroma_b.iter().all(|v| v.is_finite()),
        "chroma_b must be finite"
    );
}

#[test]
#[expect(
    clippy::cast_precision_loss,
    reason = "frame indices 0..3 are exact in f32"
)]
fn test_relief_pipeline_produces_valid_output() {
    let imgs: Vec<PlanarImage<f32>> = (0..3_i32)
        .map(|i| make_synthetic(16, 16, i as f32 * 0.1))
        .collect();
    let settings = stacker_core::settings::StackingSettings::default();
    let out = run_relief(&imgs, &settings);
    assert_eq!(out.width, 16);
    assert_eq!(out.height, 16);
    assert!(
        out.luma.iter().all(|v| v.is_finite()),
        "luma must be finite"
    );
}

/// The sRGB→linear→planar→linear→sRGB round trip returns every 8-bit code
/// **exactly**.
///
/// A known sRGB patch goes through `planar_from_dynamic` (which linearises) and
/// the shared 8-bit encoder (which de-linearises and rounds). The decode table
/// puts every code at the centre of its rounding interval, so the only error
/// left is the 1e-6-level mismatch of the matrix constants, far below half a
/// code. A tolerance of ±1 would hide a truncating encoder (white at 254) or a
/// missing linearisation (~10+ levels at midtones), so the comparison is exact.
#[test]
fn test_linearise_roundtrip() {
    use image::{DynamicImage, RgbImage};
    // A small 2×2 sRGB patch covering shadows, midtones, and highlights.
    let srgb_values: &[[u8; 3]] = &[
        [0, 0, 0],       // black
        [64, 128, 192],  // midtone mix — most sensitive to gamma errors
        [128, 128, 128], // mid-grey
        [255, 255, 255], // white
    ];

    let w = 2u32;
    let h = 2u32;
    let mut src = RgbImage::new(w, h);
    for (i, px) in src.pixels_mut().enumerate() {
        *px = image::Rgb(srgb_values[i]);
    }

    let dyn_img = DynamicImage::ImageRgb8(src);
    let planar = planar_from_dynamic(&dyn_img);
    let saving = ImageSavingSettings {
        output_format: OutputFormat::Png,
        bit_depth: 8,
        ..ImageSavingSettings::default()
    };
    let encoded = encode(&planar, &saving);
    let out = encoded.rgb8().expect("8-bit buffer");

    for (i, expected) in srgb_values.iter().enumerate() {
        assert_eq!(
            &out[i * 3..i * 3 + 3],
            expected,
            "pixel {i}: the round trip must return the original code"
        );
    }
}

/// 16-bit input values that are indistinguishable at 8-bit resolution survive
/// the load→planar→16-bit-output path **exactly**.
///
/// Two u16 grey levels in the same u8 bucket (1000 and 1010 both map to the
/// same u8) are decoded through the 65 536-entry table and written by the
/// shared 16-bit encoder (the one the tile writer uses); both must come back
/// unchanged.
#[test]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the test builds u16 codes from small in-range values"
)]
fn test_16bit_precision_preserved() {
    use image::{DynamicImage, ImageBuffer, Rgb};
    // Two grey levels that differ by 10 in u16 space but collapse to the
    // same u8 value (≈3) when quantised to 8 bits.
    let v_lo: u16 = 1_000;
    let v_hi: u16 = 1_010;

    // Sanity-check: they DO collapse at 8 bits.
    let as_u8 = |v: u16| (f32::from(v) / 65_535.0 * 255.0).round() as u8;
    assert_eq!(
        as_u8(v_lo),
        as_u8(v_hi),
        "test precondition: both values must share a u8 bucket"
    );

    // Build a 1×2 Rgb16 image: top pixel = v_lo grey, bottom = v_hi grey.
    let mut buf16: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(1, 2);
    buf16.put_pixel(0, 0, Rgb([v_lo, v_lo, v_lo]));
    buf16.put_pixel(0, 1, Rgb([v_hi, v_hi, v_hi]));
    let dyn_img = DynamicImage::ImageRgb16(buf16);

    let planar = planar_from_dynamic(&dyn_img);
    let saving = ImageSavingSettings {
        output_format: OutputFormat::Png,
        bit_depth: 16,
        ..ImageSavingSettings::default()
    };
    let encoded = encode(&planar, &saving);
    let out16 = encoded.rgb16().expect("16-bit buffer");

    // Both grey pixels come back with their exact 16-bit values.
    assert_eq!(&out16[0..3], &[v_lo; 3], "top pixel");
    assert_eq!(&out16[3..6], &[v_hi; 3], "bottom pixel");
}

#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the test writes 8-bit pixel values computed from small in-range indices"
)]
fn test_e2e_pipeline_apex_in_temp_dir() {
    use image::RgbImage;
    // Write 3 PNGs to a temp dir, run the pipeline, verify output exists and
    // has the correct dimensions.
    //
    // Images are 64×64 — large enough for akaze's internal filter kernels.
    // The pattern varies per frame so the stack is non-trivial.
    const W: u32 = 64;
    const H: u32 = 64;

    let tmp = std::env::temp_dir().join(format!("stacker_cli_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("create temp dir");

    for i in 0..3_u32 {
        let img = RgbImage::from_fn(W, H, |x, y| {
            // Vary brightness per frame so pyramids differ.
            let base = (((x + y) % 16) * 15) as u8;
            let v = base.saturating_add((i * 30) as u8);
            image::Rgb([v, v / 2, v / 3])
        });
        img.save(tmp.join(format!("frame_{i:02}.png")))
            .expect("save test frame");
    }

    let out_path = tmp.join("output.png");
    let args = CliArgs::for_test(&tmp, &out_path, "apex", 128);
    let settings = stacker_core::settings::StackingSettings::default();

    run_pipeline(&args, &settings).expect("pipeline must succeed");
    assert!(out_path.exists(), "output file must exist");

    // The pipeline crops to the area every aligned frame covers, so the
    // output may lose a few border pixels but is never larger than the
    // input and never collapses.
    let loaded = image::open(&out_path).expect("output must be readable");
    let (ow, oh) = (loaded.width(), loaded.height());
    assert!(
        ow <= W && oh <= H,
        "output {ow}x{oh} larger than input {W}x{H}"
    );
    assert!(
        ow * 4 >= W * 3 && oh * 4 >= H * 3,
        "output {ow}x{oh} cropped below 3/4 of input {W}x{H}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// A synthetic stack with detail at several scales, from single pixels up to
/// structures hundreds of pixels wide (so the coarse pyramid levels carry
/// energy), and a different in-focus region per frame (a Gaussian-weighted
/// high-frequency texture), so the winner changes across tile borders.
#[expect(
    clippy::cast_precision_loss,
    reason = "pixel coordinates and frame counts are small"
)]
fn multiscale_stack(width: usize, height: usize, count: usize) -> Vec<PlanarImage<f32>> {
    (0..count)
        .map(|k| {
            let mut img = PlanarImage::new(width, height);
            let cx = width as f32 * (k as f32 + 0.5) / count as f32;
            let cy = height as f32 * (0.5 + 0.2 * (k as f32 * 1.7).sin());
            let sigma = width as f32 / 6.0;
            for row in 0..height {
                for col in 0..width {
                    let (fx, fy) = (col as f32, row as f32);
                    let focus =
                        (-((fx - cx).powi(2) + (fy - cy).powi(2)) / (2.0 * sigma * sigma)).exp();
                    let fine = (0.9 * fx + 0.5 * fy).sin() * focus;
                    let mid = 0.5 * (0.11 * fx).sin() * (0.07 * fy).cos();
                    let coarse = 0.3 * (0.03 * fx + 0.02 * fy).cos();
                    let value = (0.5 + 0.35 * (fine + mid + coarse)).clamp(0.0, 1.0);
                    let idx = row * width + col;
                    img.luma[idx] = value;
                    img.chroma_a[idx] = (value - 0.5) * 0.2;
                    img.chroma_b[idx] = (0.5 - value) * 0.1;
                }
            }
            img
        })
        .collect()
}

/// What the CLI's fusion pass does, without the disk: cut every frame into the
/// padded tiles of `geometry`, fuse each tile with `fuse` and paste the
/// interior back.
fn fuse_in_tiles(
    frames: &[PlanarImage<f32>],
    geometry: Geometry,
    fuse: impl Fn(&[PlanarImage<f32>]) -> PlanarImage<f32>,
) -> PlanarImage<f32> {
    let (w, h) = (frames[0].width, frames[0].height);
    let mut out = PlanarImage::new(w, h);
    for cell in plan_tiles(w, h, geometry.tile_size, geometry.apron) {
        let p = &cell.padded;
        let padded: Vec<PlanarImage<f32>> = frames
            .iter()
            .map(|f| extract_tile(f, p.start_x, p.start_y, p.width, p.height))
            .collect();
        let fused = fuse(&padded);
        let (ix, iy) = cell.interior_offset;
        let c = &cell.interior;
        paste_interior(
            &mut out, &fused, ix, iy, c.start_x, c.start_y, c.width, c.height,
        );
    }
    out
}

fn max_plane_error(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> [f32; 3] {
    let worst = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .fold(0.0_f32, |acc, (p, q)| acc.max((p - q).abs()))
    };
    [
        worst(&a.luma, &b.luma),
        worst(&a.chroma_a, &b.chroma_a),
        worst(&a.chroma_b, &b.chroma_b),
    ]
}

/// **Tiled `Apex` equals whole-image `Apex`** (TST-2, CORE-3).
///
/// The image is 128×128 with 32-px tiles, so the grid is 4×4 and the padded
/// region of an inner tile (interior plus the 32-px apron derived from the 4
/// pyramid levels) is a strict sub-rectangle of the image: unlike the former
/// 64×64 / 16-px version, in which every padded tile was the whole image, the
/// seams of a too-small apron would show here. The tolerance is the review's
/// 0.005 per plane.
#[test]
fn test_tiled_matches_whole_image_apex() {
    const TOL: f32 = 0.005;
    let settings = StackingSettings {
        pyramid_levels: 4,
        ..StackingSettings::default()
    };
    let geometry = Geometry::choose(32, Method::Apex, &settings);
    assert_eq!((geometry.tile_size, geometry.apron), (32, 32));
    let frames = multiscale_stack(128, 128, 4);

    // The padded tile of an inner cell is smaller than the image.
    let cells = plan_tiles(128, 128, geometry.tile_size, geometry.apron);
    assert_eq!(cells.len(), 16);
    assert!(
        cells
            .iter()
            .all(|c| c.padded.width < 128 || c.padded.height < 128),
        "every padded tile must be a strict part of the image"
    );

    let whole = run_apex(&frames, &settings);
    let tiled = fuse_in_tiles(&frames, geometry, |t| run_apex(t, &settings));
    let [luma, chroma_a, chroma_b] = max_plane_error(&tiled, &whole);
    assert!(luma <= TOL, "tiled luma deviates by {luma:.6} (tol={TOL})");
    assert!(
        chroma_a <= TOL,
        "tiled chroma_a deviates by {chroma_a:.6} (tol={TOL})"
    );
    assert!(
        chroma_b <= TOL,
        "tiled chroma_b deviates by {chroma_b:.6} (tol={TOL})"
    );
}

/// The same at the default 8 levels, where the derived apron is 512 px: a 1024²
/// image with 256-px tiles is the review's proving case for CORE-3.
#[test]
#[ignore = "heavy: 1024x1024 Apex at the default 8 levels, tiled and whole"]
fn test_tiled_matches_whole_image_apex_default_levels() {
    const TOL: f32 = 0.005;
    let settings = StackingSettings::default();
    let geometry = Geometry::choose(256, Method::Apex, &settings);
    assert_eq!(geometry.apron, 512);
    let frames = multiscale_stack(1024, 1024, 3);
    let whole = run_apex(&frames, &settings);
    let tiled = fuse_in_tiles(&frames, geometry, |t| run_apex(t, &settings));
    let [luma, ..] = max_plane_error(&tiled, &whole);
    assert!(luma <= TOL, "tiled luma deviates by {luma:.6} (tol={TOL})");
}

/// **The tiled output writer encodes exactly like the whole-image encoder.**
///
/// One fused image is encoded twice: by `encode` in one go, and by pasting its
/// 16-px tiles into a zeroed buffer with `paste_tile` (the CLI's streaming
/// path). The only thing that differs is the addressing, so the bytes must be
/// identical (tolerance 0), at 8 and at 16 bits.
#[test]
fn test_streamed_tile_output_matches_whole_image_encoding() {
    const W: usize = 64;
    const H: usize = 64;
    const TILE: usize = 16;

    let settings = StackingSettings::default();
    let whole_fused = run_apex(&multiscale_stack(W, H, 4), &settings);

    for bit_depth in [8, 16] {
        let saving = ImageSavingSettings {
            output_format: OutputFormat::Png,
            bit_depth,
            ..ImageSavingSettings::default()
        };
        let reference = encode(&whole_fused, &saving);

        let mut streamed = EncodedImage::zeroed(W, H, OutputFormat::Png, bit_depth == 16, 95);
        for coord in enumerate_tiles(W, H, TILE) {
            // The "padded tile" is the whole fused image here, so the interior
            // offset is the tile's own start.
            streamed.paste_tile(
                &whole_fused,
                (coord.start_x, coord.start_y),
                (coord.start_x, coord.start_y, coord.width, coord.height),
            );
        }

        assert_eq!(streamed.is_16_bit(), bit_depth == 16);
        assert_eq!(streamed.rgb8(), reference.rgb8(), "8-bit samples differ");
        assert_eq!(streamed.rgb16(), reference.rgb16(), "16-bit samples differ");
        assert!(
            reference.rgb8().is_some() || reference.rgb16().is_some(),
            "the reference holds samples"
        );
    }
}

// ── Image-quality metric helpers ──────────────────────────────────────────
//
// These are used by the fast unit tests (which validate the metrics on
// synthetic data) and by the heavy real-image quality tests (which run the
// full pipeline on the sample stack and compare against zerene.jpg).
//
// All helpers operate on grayscale `f32` pixel slices in [0, 1].

/// **Tenengrad sharpness**: mean squared Sobel gradient magnitude.
///
/// Applies a 3×3 Sobel operator in X and Y, accumulates (Gx² + Gy²) for
/// every interior pixel, and returns the mean over all interior pixels.
/// Higher = sharper.
///
/// Note: ghosting/misalignment ADDS high-frequency energy, so a broken
/// stack can score higher than a clean reference.  This metric is therefore
/// only meaningful in combination with SSIM (see `ssim`).
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "the Sobel neighbours are indexed with signed offsets inside the interior, and the count is exact in f64"
)]
fn tenengrad(gray: &[f32], width: usize, height: usize) -> f64 {
    // Sobel kernels:
    //   Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
    //   Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
    let mut sum = 0.0_f64;
    let mut count = 0u64;
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // SAFETY-note: y and x are in [1, dim-2], and dy/dx are ±1,
            // so ny/nx are always in [0, dim-1].  The casts are intentional.
            let s = |dy: isize, dx: isize| -> f64 {
                let ny = (y as isize + dy) as usize;
                let nx = (x as isize + dx) as usize;
                f64::from(gray[ny * width + nx])
            };
            let gx = -s(-1, -1) + s(-1, 1) - 2.0 * s(0, -1) + 2.0 * s(0, 1) - s(1, -1) + s(1, 1);
            let gy = -s(-1, -1) - 2.0 * s(-1, 0) - s(-1, 1) + s(1, -1) + 2.0 * s(1, 0) + s(1, 1);
            sum += gx * gx + gy * gy;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// **Windowed SSIM** (Structural Similarity Index Measure).
///
/// Computes SSIM over non-overlapping 8×8 windows and returns the mean.
/// Images `a` and `b` must be the same size.
///
/// Constants follow the original Wang et al. paper:
///   L = 1.0 (pixel range normalised to [0,1])
///   C1 = (0.01 * L)² = 1e-4
///   C2 = (0.03 * L)² = 9e-4
///
/// Return value is in [−1, 1]; 1.0 means identical images.
#[expect(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::suspicious_operation_groupings,
    reason = "the SSIM formula of Wang et al. uses the paper's a/b, mu, var and cov names"
)]
fn ssim(a: &[f32], b: &[f32], width: usize, height: usize) -> f64 {
    const WIN: usize = 8;
    const C1: f64 = 1e-4; // (0.01 * 1.0)^2
    const C2: f64 = 9e-4; // (0.03 * 1.0)^2

    let mut total = 0.0_f64;
    let mut n_windows = 0u32;

    let y_wins = height / WIN;
    let x_wins = width / WIN;

    for wy in 0..y_wins {
        for wx in 0..x_wins {
            let mut sum_a = 0.0_f64;
            let mut sum_b = 0.0_f64;
            let mut sum_aa = 0.0_f64;
            let mut sum_bb = 0.0_f64;
            let mut sum_ab = 0.0_f64;
            let n = (WIN * WIN) as f64;

            for dy in 0..WIN {
                for dx in 0..WIN {
                    let row = wy * WIN + dy;
                    let col = wx * WIN + dx;
                    let av = f64::from(a[row * width + col]);
                    let bv = f64::from(b[row * width + col]);
                    sum_a += av;
                    sum_b += bv;
                    sum_aa += av * av;
                    sum_bb += bv * bv;
                    sum_ab += av * bv;
                }
            }

            let mu_a = sum_a / n;
            let mu_b = sum_b / n;
            // Variance = E[X²] − (E[X])²
            let var_a = (sum_aa / n) - (mu_a * mu_a);
            let var_b = (sum_bb / n) - (mu_b * mu_b);
            let cov = (sum_ab / n) - (mu_a * mu_b);

            let num = (2.0 * mu_a * mu_b + C1) * (2.0 * cov + C2);
            let den = (mu_a * mu_a + mu_b * mu_b + C1) * (var_a + var_b + C2);
            total += num / den;
            n_windows += 1;
        }
    }

    if n_windows == 0 {
        1.0
    } else {
        total / f64::from(n_windows)
    }
}

/// **Coarse integer alignment** via zero-mean normalised cross-correlation (ZNCC).
///
/// Downscales both images to at most `coarse_w` pixels wide (preserving
/// aspect), then sweeps a search window of `±max_shift` pixels, evaluating
/// ZNCC at each integer (dx, dy) offset.  Returns the shift `(dx, dy)` that
/// maximises ZNCC, in **original-image pixels** (scaled back up).
///
/// For each candidate shift the mean of each image is recomputed over the
/// shared overlap region so that the normalisation is correct regardless of
/// border effects.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "pixel coordinates and shifts are small signed integers that are scaled to and from f32/f64"
)]
fn coarse_align_shift(
    a: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
    max_shift: usize,
    coarse_w: usize,
) -> (i32, i32) {
    // Downscale factor (integer for speed; we only need coarse alignment).
    let scale = (width / coarse_w).max(1);
    let cw = width / scale;
    let ch = height / scale;
    let ms = (max_shift / scale).max(1);

    // Box-average downscale of both images.
    let downscale = |src: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0_f32; cw * ch];
        for cy in 0..ch {
            for cx in 0..cw {
                let mut s = 0.0_f32;
                let mut cnt = 0u32;
                for dy in 0..scale {
                    for dx in 0..scale {
                        let sy = cy * scale + dy;
                        let sx = cx * scale + dx;
                        if sy < height && sx < width {
                            s += src[sy * width + sx];
                            cnt += 1;
                        }
                    }
                }
                out[cy * cw + cx] = if cnt > 0 { s / cnt as f32 } else { 0.0 };
            }
        }
        out
    };

    let da = downscale(a);
    let db = downscale(b);

    // Leave a border of `ms` pixels so every searched shift keeps the
    // evaluation window inside both images.
    let border = ms + 1;
    if ch <= 2 * border || cw <= 2 * border {
        return (0, 0);
    }
    let y0 = border;
    let y1 = ch - border;
    let x0 = border;
    let x1 = cw - border;

    // Search over ±ms shifts; for each shift recompute the per-overlap mean
    // so the ZNCC numerator and denominator are correctly normalised.
    let mut best_zncc = f64::NEG_INFINITY;
    // best_shift[0]=dx, best_shift[1]=dy; array avoids similar-names lint.
    let mut best_shift = [0i32; 2];

    let ms_i = ms as i32;
    for sdelta_y in -ms_i..=ms_i {
        for sdelta_x in -ms_i..=ms_i {
            // Compute overlap-region means for this shift.
            let mut sum_a = 0.0_f64;
            let mut sum_b = 0.0_f64;
            let mut n = 0u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let by = (y as i32 + sdelta_y) as usize;
                    let bx = (x as i32 + sdelta_x) as usize;
                    if by >= ch || bx >= cw {
                        continue;
                    }
                    sum_a += f64::from(da[y * cw + x]);
                    sum_b += f64::from(db[by * cw + bx]);
                    n += 1;
                }
            }
            if n == 0 {
                continue;
            }
            let mu_a = sum_a / n as f64;
            let mu_b = sum_b / n as f64;

            // ZNCC over the same overlap.
            let mut num = 0.0_f64;
            let mut ss_a = 0.0_f64;
            let mut ss_b = 0.0_f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let by = (y as i32 + sdelta_y) as usize;
                    let bx = (x as i32 + sdelta_x) as usize;
                    if by >= ch || bx >= cw {
                        continue;
                    }
                    let av = f64::from(da[y * cw + x]) - mu_a;
                    let bv = f64::from(db[by * cw + bx]) - mu_b;
                    num += av * bv;
                    ss_a += av * av;
                    ss_b += bv * bv;
                }
            }
            let denom = ss_a.sqrt() * ss_b.sqrt();
            let zncc = if denom > 1e-12 { num / denom } else { 0.0 };
            if zncc > best_zncc {
                best_zncc = zncc;
                best_shift = [sdelta_x, sdelta_y];
            }
        }
    }

    // Scale back to original-image pixels.
    (best_shift[0] * scale as i32, best_shift[1] * scale as i32)
}

/// Convert a grayscale `image::GrayImage` to a `Vec<f32>` in [0, 1].
fn gray_image_to_f32(img: &image::GrayImage) -> Vec<f32> {
    img.pixels().map(|p| f32::from(p[0]) / 255.0).collect()
}

/// Resize a `DynamicImage` to a given width, maintaining aspect ratio.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the rounded target height is a positive pixel count"
)]
fn resize_to_width(img: &image::DynamicImage, target_w: u32) -> image::DynamicImage {
    let (w, h) = img.dimensions();
    if w == 0 {
        return img.clone();
    }
    let target_h = (f64::from(h) * f64::from(target_w) / f64::from(w)).round() as u32;
    image::DynamicImage::ImageRgba8(image::imageops::resize(
        img,
        target_w,
        target_h,
        image::imageops::FilterType::Lanczos3,
    ))
}

// ── Fast metric unit tests (always run, no #[ignore]) ────────────────────

/// SSIM of an image with itself must be ≈ 1.0.
/// SSIM of an image with a blurred copy must be < 0.9.
/// SSIM of an image with pure noise must be < 0.5.
#[test]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a 64x64 synthetic image: indices are exact in f32 and the 8-bit codes are clamped"
)]
fn test_ssim_properties() {
    const W: usize = 64;
    const H: usize = 64;

    // Build a deterministic synthetic image (checkerboard + gradient).
    let mut img = vec![0.0_f32; W * H];
    for y in 0..H {
        for x in 0..W {
            img[y * W + x] = (((x + y) % 8) as f32 / 8.0) + (x as f32 / W as f32) * 0.3;
        }
    }
    // Clamp to [0,1].
    for v in &mut img {
        *v = v.clamp(0.0, 1.0);
    }

    // Self SSIM ≈ 1.0.
    let self_ssim = ssim(&img, &img, W, H);
    assert!(
        (self_ssim - 1.0).abs() < 1e-6,
        "ssim(x, x) = {self_ssim:.6}, expected ≈ 1.0"
    );

    // Heavily blurred: convert to DynamicImage, blur, convert back.
    let gray_u8: image::GrayImage = image::GrayImage::from_fn(W as u32, H as u32, |x, y| {
        image::Luma([(img[y as usize * W + x as usize] * 255.0) as u8])
    });
    let blurred_rgba =
        image::imageops::blur(&image::DynamicImage::ImageLuma8(gray_u8).to_rgba8(), 8.0);
    let blurred: Vec<f32> = blurred_rgba
        .pixels()
        .map(|p| f32::from(p[0]) / 255.0)
        .collect();
    let blur_ssim = ssim(&img, &blurred, W, H);
    assert!(
        blur_ssim < 0.9,
        "ssim(x, blurred_x) = {blur_ssim:.4}, expected < 0.9"
    );

    // Pure noise: use a simple LCG seeded deterministically.
    let mut noise = vec![0.0_f32; W * H];
    let mut rng: u64 = 0x_dead_beef_cafe_1337;
    for v in &mut noise {
        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *v = ((rng >> 33) as f32) / (u32::MAX as f32);
    }
    let noise_ssim = ssim(&img, &noise, W, H);
    assert!(
        noise_ssim < 0.5,
        "ssim(x, noise) = {noise_ssim:.4}, expected < 0.5"
    );
}

/// Tenengrad of a sharp synthetic image must exceed that of a blurred copy.
///
/// We use a step-edge pattern (left half = 0, right half = 1) rather than
/// alternating columns.  A period-2 column pattern aliases into the Sobel
/// null-space (all neighbours at x±1 are the same value as the reference
/// side of the kernel), so the Sobel response is zero for that pattern even
/// though it is "sharp" in a Nyquist sense.  A step edge has a large,
/// localised Sobel response at the boundary and is therefore a correct test
/// for the `tenengrad` implementation.
#[test]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a 64x64 synthetic image whose 8-bit codes are clamped"
)]
fn test_tenengrad_sharp_beats_blurred() {
    const W: usize = 64;
    const H: usize = 64;

    // Sharp: step edge at the midpoint — left half = 0.0, right half = 1.0.
    let sharp: Vec<f32> = (0..W * H)
        .map(|i| if (i % W) >= W / 2 { 1.0_f32 } else { 0.0_f32 })
        .collect();

    // Blurred: convert to GrayImage, blur with σ=5, convert back.
    let gray_u8 = image::GrayImage::from_fn(W as u32, H as u32, |x, y| {
        image::Luma([(sharp[y as usize * W + x as usize] * 255.0) as u8])
    });
    let blurred_rgba =
        image::imageops::blur(&image::DynamicImage::ImageLuma8(gray_u8).to_rgba8(), 5.0);
    let blurred: Vec<f32> = blurred_rgba
        .pixels()
        .map(|p| f32::from(p[0]) / 255.0) // R channel of RGBA (grey image)
        .collect();

    let t_sharp = tenengrad(&sharp, W, H);
    let t_blurred = tenengrad(&blurred, W, H);
    assert!(
        t_sharp > t_blurred,
        "tenengrad(sharp)={t_sharp:.6} must exceed tenengrad(blurred)={t_blurred:.6}"
    );
}

/// `coarse_align_shift` must recover a known small synthetic translation.
///
/// The shift values are chosen to be exact multiples of the downscale factor
/// (scale = `W/coarse_w` = 128/128 = 1, i.e. no downscaling) so that the
/// integer grid in the search space aligns perfectly with the true shift.
/// Using `coarse_w >= W` forces `scale = 1` and avoids the half-pixel
/// aliasing that occurs when the shift is not divisible by the scale factor.
#[test]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "a 128x128 synthetic image with a known small signed shift"
)]
fn test_coarse_align_shift_recovers_known_shift() {
    const W: usize = 128;
    const H: usize = 128;
    const TRUE_DX: i32 = 5;
    const TRUE_DY: i32 = -3;
    // coarse_w >= W → scale = 1 → no downscaling → exact integer shifts.
    const COARSE_W: usize = 256;

    // Build a reference image with a structured spatial pattern (not constant,
    // not periodic with period ≤ max_shift so ZNCC has a clear global maximum).
    let mut a = vec![0.0_f32; W * H];
    for y in 0..H {
        for x in 0..W {
            a[y * W + x] =
                (((x * 7 + y * 13) % 31) as f32 / 30.0) * ((x as f32 * 0.3).sin().abs() + 0.1);
        }
    }

    // Build b as `a` shifted by (TRUE_DX, TRUE_DY): b[y][x] = a[y-dy][x-dx].
    // Zero-fill border pixels where `a` has no data.
    let mut b = vec![0.0_f32; W * H];
    for y in 0..H {
        for x in 0..W {
            let sy = y as i32 - TRUE_DY;
            let sx = x as i32 - TRUE_DX;
            // Bounds check is intentional: sy/sx may be negative.
            if sy >= 0 && sy < H as i32 && sx >= 0 && sx < W as i32 {
                b[y * W + x] = a[sy as usize * W + sx as usize];
            }
        }
    }

    // With coarse_w=256 > W=128, scale=max(1, 128/256)=1.
    // max_shift=16 → ±16 px search, well within the image.
    let (dx, dy) = coarse_align_shift(&a, &b, W, H, 16, COARSE_W);
    assert!(
        (dx - TRUE_DX).abs() <= 1 && (dy - TRUE_DY).abs() <= 1,
        "coarse_align_shift returned ({dx},{dy}), expected ({TRUE_DX},{TRUE_DY}) ±1"
    );
}

// ── Real-image quality test (full-resolution; heavy) ─────────────────────
//
// Runs the entire pipeline on the 28-frame 3840×2160 sample stack and
// compares against zerene.jpg using the metric helpers above.
//
// Thresholds are intentionally conservative to allow the initial calibration
// run to succeed; tighten them once observed values are known.

/// Threshold constants — adjust after seeing actual measured values.
///
/// `SHARPNESS_MIN_RATIO`: stackr tenengrad must be at least this fraction of
///   zerene tenengrad.  A real stack should be at least as sharp as zerene.
const SHARPNESS_MIN_RATIO: f64 = 0.85;
/// `SSIM_MIN`: windowed SSIM between stackr and zerene (after coarse alignment).
///   Ghosting / comb artefacts produce low SSIM; a clean result should exceed
///   this.  Calibrated against observed values:
///   - full-res / ≥1280px inputs:  stackr vs zerene SSIM ≈ 0.45–0.65 → gate at 0.40
///   - <1280px inputs: cross-resolution comparison drags SSIM down to ~0.35–0.52;
///     use a looser floor of 0.30 (still catches badly ghosted/misaligned results).
const SSIM_MIN: f64 = 0.40;
/// Looser SSIM gate for heavily downscaled inputs (<1280 px wide) where the
/// stackr-vs-zerene comparison is inherently cross-resolution.  Observed 0.385
/// on 4-frame 160px inputs; 0.30 catches genuine ghosting while passing clean results.
const SSIM_MIN_LOWRES: f64 = 0.30;
/// `BEATS_SINGLE_FRAME`: stackr tenengrad must be at least this fraction of the
///   sharpest single input frame's tenengrad.  A focus stack covers all focal
///   planes, so it should be sharper than any single frame.
const BEATS_SINGLE_FRAME: f64 = 1.00;
/// Width used when comparing sharpness (resize both to this width).
const SHARPNESS_COMPARE_W: u32 = 1920;
/// Width used for SSIM comparison (smaller = faster; 1280 is plenty).
const SSIM_COMPARE_W: u32 = 1280;
/// Tile size for the real-image pipeline run.
const QUALITY_TILE_SIZE: usize = 512;
/// Max shift searched by `coarse_align_shift` (original-image pixels).
const COARSE_MAX_SHIFT: usize = 64;
/// Coarse-search downscale target width.
const COARSE_SEARCH_W: usize = 320;

#[test]
#[ignore = "heavy: runs full pipeline on sample images (28×~8MP + AKAZE; use downscaled variant for CI)"]
fn test_quality_vs_zerene_full_res() {
    quality_vs_zerene_impl(None, None);
}

/// Same gate as `test_quality_vs_zerene_full_res` but pre-downscales every
/// input frame to 1280-wide before the pipeline so the run takes minutes
/// instead of the full-resolution hour.  Numbers are from downscaled inputs:
/// sharpness values are lower than full-res but ratios and SSIM stay
/// comparably meaningful.
#[test]
#[ignore = "heavy: 28 frames at 1280px; ~5 min; run manually with --ignored"]
fn test_quality_vs_zerene_downscaled() {
    quality_vs_zerene_impl(Some(1280), None);
}

/// Ultra-light calibration variant: 2 evenly-spaced frames downscaled to
/// 160 px wide.  Metrics reflect that tiny input and are useful only for
/// threshold calibration, not as a quality bar.
#[test]
#[ignore = "quality calibration on sample images; run manually with --ignored"]
fn test_quality_vs_zerene_quick() {
    quality_vs_zerene_impl(Some(160), Some(2));
}

/// Mid-tier calibration: 4 evenly-spaced frames at 160 px wide, covering
/// more focal planes than the 2-frame quick variant.
#[test]
#[ignore = "quality calibration on sample images; run manually with --ignored"]
fn test_quality_vs_zerene_4frame() {
    quality_vs_zerene_impl(Some(160), Some(4));
}

/// Shared implementation for all quality test variants.
///
/// * `downscale_inputs_to` — if `Some(w)`, each input frame is downscaled
///   to width `w` before being fed to the pipeline.  `None` = full res.
/// * `max_frames` — if `Some(n)`, subsample the sorted frame list to at
///   most `n` frames (evenly spaced), so AKAZE finishes faster.  `None`
///   = use all frames.
#[expect(
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "one linear calibration script: downscaled sizes are small positive pixel counts, and splitting it would hide the pipeline-then-metrics order"
)]
fn quality_vs_zerene_impl(downscale_inputs_to: Option<u32>, max_frames: Option<usize>) {
    // ── 1. Locate sample directory ────────────────────────────────────────
    // CARGO_MANIFEST_DIR = <workspace>/apps/stacker-cli
    // sample dir         = <workspace>/sample  (two levels up)
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let sample_dir = manifest_dir.join("../../sample");
    let zerene_path = sample_dir.join("result/zerene.jpg");

    if !sample_dir.exists() {
        eprintln!(
            "[quality test] SKIP: sample dir not found at {}",
            sample_dir.display()
        );
        return;
    }
    if !zerene_path.exists() {
        eprintln!(
            "[quality test] SKIP: zerene.jpg not found at {}",
            zerene_path.display()
        );
        return;
    }

    // Collect input frames (non-recursive, same logic as pipeline).
    let mut frame_paths: Vec<std::path::PathBuf> = std::fs::read_dir(&sample_dir)
        .expect("read sample dir")
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && matches!(
                    p.extension()
                        .and_then(|e| e.to_str())
                        .map(str::to_ascii_lowercase)
                        .as_deref(),
                    Some("jpg" | "jpeg" | "png" | "tif" | "tiff")
                )
        })
        .collect();
    frame_paths.sort();

    if frame_paths.is_empty() {
        eprintln!(
            "[quality test] SKIP: no input frames found in {}",
            sample_dir.display()
        );
        return; // Not a test failure; sample data simply not present.
    }

    // Optionally subsample the frame list to speed up AKAZE.
    if let Some(n) = max_frames
        && frame_paths.len() > n
    {
        let total = frame_paths.len();
        // Pick n evenly-spaced indices across [0, total).
        frame_paths = (0..n)
            .map(|i| i * (total - 1) / (n - 1).max(1))
            .map(|i| frame_paths[i].clone())
            .collect();
    }
    eprintln!("[quality test] {} input frames selected", frame_paths.len());

    // ── 2. Optionally downscale inputs into a temp dir ────────────────────
    // tmp_cleanup holds the temp dir path so it can be deleted at the end.
    let tmp_cleanup: Option<std::path::PathBuf>;
    let pipeline_input_dir: std::path::PathBuf;
    if let Some(target_w) = downscale_inputs_to {
        let tmp = std::env::temp_dir().join(format!(
            "stacker_quality_ds_{}_w{}_f{}",
            std::process::id(),
            target_w,
            max_frames.unwrap_or(0),
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("create downscale temp dir");

        eprintln!(
            "[quality test] downscaling {} frames to width={target_w} ...",
            frame_paths.len()
        );
        for path in &frame_paths {
            let img =
                image::open(path).unwrap_or_else(|e| panic!("open frame {}: {e}", path.display()));
            let ds = resize_to_width(&img, target_w);
            let name = path.file_name().expect("frame has filename");
            // Save as PNG to avoid re-compression artefacts.
            let out_name = format!(
                "{}.png",
                std::path::Path::new(name)
                    .file_stem()
                    .unwrap_or(name)
                    .to_string_lossy()
            );
            ds.save(tmp.join(&out_name)).expect("save downscaled frame");
        }
        pipeline_input_dir = tmp.clone();
        tmp_cleanup = Some(tmp);
    } else {
        pipeline_input_dir = sample_dir;
        tmp_cleanup = None;
    }

    // ── 3. Get expected output dimensions ────────────────────────────────
    // Load the first frame to get expected W×H.
    let first_frame_path = if downscale_inputs_to.is_some() {
        // First PNG in tmp dir.
        let mut ps: Vec<_> = std::fs::read_dir(&pipeline_input_dir)
            .expect("read ds dir")
            .filter_map(std::result::Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
            .collect();
        ps.sort();
        ps.into_iter()
            .next()
            .expect("at least one downscaled frame")
    } else {
        frame_paths[0].clone()
    };
    let first_img =
        image::open(&first_frame_path).unwrap_or_else(|e| panic!("open first frame: {e}"));
    let expected_w = first_img.width();
    let expected_h = first_img.height();
    eprintln!("[quality test] expected output size: {expected_w}×{expected_h}");

    // ── 4. Run pipeline ───────────────────────────────────────────────────
    // One directory per process *and* variant: several variants may run
    // concurrently in one test process.
    let variant_tag = format!(
        "ds{}_n{}",
        downscale_inputs_to.unwrap_or(0),
        max_frames.unwrap_or(0)
    );
    let out_dir = std::env::temp_dir().join(format!(
        "stacker_quality_out_{}_{variant_tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&out_dir).expect("create output temp dir");
    let out_path = out_dir.join("stackr_apex.png");

    let args = CliArgs::for_test(&pipeline_input_dir, &out_path, "apex", QUALITY_TILE_SIZE);
    let settings = stacker_core::settings::StackingSettings::default();

    eprintln!("[quality test] running pipeline (mode=apex, tile_size={QUALITY_TILE_SIZE}) ...");
    run_pipeline(&args, &settings).expect("pipeline must succeed on sample images");
    assert!(out_path.exists(), "pipeline output file must exist");

    // ── 5. Load stackr output + zerene ───────────────────────────────────
    let stackr_img = image::open(&out_path).unwrap_or_else(|e| panic!("open stackr output: {e}"));
    let zerene_img = image::open(&zerene_path).unwrap_or_else(|e| panic!("open zerene: {e}"));

    // ── 6. Size sanity ────────────────────────────────────────────────────
    // The pipeline crops the output to the area covered by every aligned
    // frame, so the result may be slightly smaller than the input but never
    // larger, and never collapsed (a crop below three quarters of either
    // dimension means the alignment ran away).
    let stackr_w = stackr_img.width();
    let stackr_h = stackr_img.height();
    eprintln!("[quality test] stackr output size:   {stackr_w}×{stackr_h}");
    eprintln!(
        "[quality test] zerene  output size:  {}×{}",
        zerene_img.width(),
        zerene_img.height()
    );
    assert!(
        stackr_w <= expected_w && stackr_h <= expected_h,
        "stackr output {stackr_w}×{stackr_h} larger than input {expected_w}×{expected_h}"
    );
    assert!(
        stackr_w * 4 >= expected_w * 3 && stackr_h * 4 >= expected_h * 3,
        "stackr output {stackr_w}×{stackr_h} cropped below 3/4 of input {expected_w}×{expected_h}"
    );

    // ── 7. Sharpness comparison (Tenengrad) ───────────────────────────────
    // Resize both to SHARPNESS_COMPARE_W for a fair comparison.
    let stackr_for_sharp = resize_to_width(&stackr_img, SHARPNESS_COMPARE_W);
    let zerene_for_sharp = resize_to_width(&zerene_img, SHARPNESS_COMPARE_W);

    let sw = stackr_for_sharp.width() as usize;
    let sh = stackr_for_sharp.height() as usize;
    let zw = zerene_for_sharp.width() as usize;
    let zh = zerene_for_sharp.height() as usize;

    let stackr_gray_sharp = stackr_for_sharp.to_luma8();
    let zerene_gray_sharp = zerene_for_sharp.to_luma8();
    let stackr_s: Vec<f32> = gray_image_to_f32(&stackr_gray_sharp);
    let zerene_s: Vec<f32> = gray_image_to_f32(&zerene_gray_sharp);

    let t_stackr = tenengrad(&stackr_s, sw, sh);
    let t_zerene = tenengrad(&zerene_s, zw, zh);
    let sharp_ratio = if t_zerene > 0.0 {
        t_stackr / t_zerene
    } else {
        1.0
    };
    eprintln!("[quality test] tenengrad(stackr @{SHARPNESS_COMPARE_W}w) = {t_stackr:.4}");
    eprintln!("[quality test] tenengrad(zerene @{SHARPNESS_COMPARE_W}w) = {t_zerene:.4}");
    eprintln!("[quality test] sharpness ratio (stackr/zerene)            = {sharp_ratio:.4}");
    // Only assert sharpness when the inputs were not heavily downscaled
    // (a 480px stackr vs 3840px zerene comparison is unfair: zerene is sharper
    // by construction).  The sharpness gate is meaningful for full-res or
    // ≥1280px downscaled inputs where both have comparable source resolution.
    let inputs_comparable_res = downscale_inputs_to.is_none_or(|w| w >= 1280);
    if inputs_comparable_res {
        assert!(
            sharp_ratio >= SHARPNESS_MIN_RATIO,
            "stackr tenengrad ({t_stackr:.4}) is < {SHARPNESS_MIN_RATIO}× zerene ({t_zerene:.4}); ratio={sharp_ratio:.4}"
        );
    } else {
        eprintln!(
            "[quality test] (sharpness gate skipped: inputs downscaled to {}px < 1280px)",
            downscale_inputs_to.unwrap_or(0)
        );
    }

    // ── 8. Best single-frame sharpness ────────────────────────────────────
    // Measure every input frame (downscaled consistently) and take the max.
    let input_frame_list: Vec<std::path::PathBuf> = if downscale_inputs_to.is_some() {
        let mut ps: Vec<_> = std::fs::read_dir(&pipeline_input_dir)
            .expect("read ds dir")
            .filter_map(std::result::Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                matches!(
                    p.extension()
                        .and_then(|e| e.to_str())
                        .map(str::to_ascii_lowercase)
                        .as_deref(),
                    Some("jpg" | "jpeg" | "png" | "tif" | "tiff")
                )
            })
            .collect();
        ps.sort();
        ps
    } else {
        frame_paths.clone()
    };

    let mut best_single_t = 0.0_f64;
    for path in &input_frame_list {
        let img = image::open(path).unwrap_or_else(|e| panic!("open input frame: {e}"));
        let ds = resize_to_width(&img, SHARPNESS_COMPARE_W);
        let gray = ds.to_luma8();
        let pix: Vec<f32> = gray_image_to_f32(&gray);
        let t = tenengrad(&pix, ds.width() as usize, ds.height() as usize);
        if t > best_single_t {
            best_single_t = t;
        }
    }
    eprintln!("[quality test] tenengrad(best single input frame) = {best_single_t:.4}");
    assert!(
        t_stackr >= BEATS_SINGLE_FRAME * best_single_t,
        "stackr tenengrad ({t_stackr:.4}) < {BEATS_SINGLE_FRAME}× best single frame ({best_single_t:.4}); stack should be sharper everywhere"
    );

    // ── 9. Structural fidelity (SSIM with coarse alignment) ───────────────
    // Downscale both to SSIM_COMPARE_W.
    let stackr_for_ssim = resize_to_width(&stackr_img, SSIM_COMPARE_W);
    let zerene_for_ssim = resize_to_width(&zerene_img, SSIM_COMPARE_W);

    let ssim_w = stackr_for_ssim.width() as usize;
    let ssim_h = stackr_for_ssim.height() as usize;

    let sg = stackr_for_ssim.to_luma8();
    let zg = zerene_for_ssim.to_luma8();
    let spx: Vec<f32> = gray_image_to_f32(&sg);
    let zpx_full: Vec<f32> = gray_image_to_f32(&zg);

    // Coarse-align stackr → zerene; zerene may be slightly offset.
    let (dx, dy) = coarse_align_shift(
        &spx,
        &zpx_full,
        ssim_w,
        ssim_h,
        COARSE_MAX_SHIFT,
        COARSE_SEARCH_W,
    );
    eprintln!("[quality test] coarse_align_shift(stackr→zerene): dx={dx}, dy={dy}");

    // Compute SSIM over the overlapping region after applying the shift.
    // We crop both images to their common overlap.
    let ox = dx.unsigned_abs() as usize;
    let oy = dy.unsigned_abs() as usize;
    let overlap_w = ssim_w.saturating_sub(ox);
    let overlap_h = ssim_h.saturating_sub(oy);

    // Stackr crop: starts at (ox,oy) if dx<0 else (0,0) — but we shift b
    // by (dx,dy) onto a, so take the region that is valid in both.
    // Crop origin in stackr: if shift is negative, stackr starts after the border.
    // Crop origin in zerene: if shift is positive, zerene starts after the border.
    // sa_*/sb_* are intentionally similar names (a=stackr, b=zerene crop origins).
    let stackr_crop_x = if dx < 0 { ox } else { 0 };
    let stackr_crop_y = if dy < 0 { oy } else { 0 };
    let zerene_crop_x = if dx > 0 { ox } else { 0 };
    let zerene_crop_y = if dy > 0 { oy } else { 0 };

    let crop =
        |pix: &[f32], src_w: usize, sx: usize, sy: usize, cw: usize, ch: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(cw * ch);
            for row in 0..ch {
                let start = (sy + row) * src_w + sx;
                out.extend_from_slice(&pix[start..start + cw]);
            }
            out
        };

    // Ensure zerene is the same pixel dimensions as stackr at SSIM scale.
    // zerene_for_ssim was resized by width; if zerene has a different aspect
    // ratio it may differ in height by ±1 px.  Re-resize to exact match.
    let zerene_dims = (
        zerene_for_ssim.width() as usize,
        zerene_for_ssim.height() as usize,
    );
    let zpx: Vec<f32> = if zerene_dims == (ssim_w, ssim_h) {
        // Fast path: dimensions already match — use the pre-computed grayscale.
        zpx_full
    } else {
        // Zerene has a different aspect — resize to match stackr exactly.
        let zr_exact = image::imageops::resize(
            &zerene_for_ssim.to_rgba8(),
            ssim_w as u32,
            ssim_h as u32,
            image::imageops::FilterType::Lanczos3,
        );
        zr_exact.pixels().map(|p| f32::from(p[0]) / 255.0).collect()
    };

    let crop_s = crop(
        &spx,
        ssim_w,
        stackr_crop_x,
        stackr_crop_y,
        overlap_w,
        overlap_h,
    );
    let crop_z = crop(
        &zpx,
        ssim_w,
        zerene_crop_x,
        zerene_crop_y,
        overlap_w,
        overlap_h,
    );

    let quality_ssim = ssim(&crop_s, &crop_z, overlap_w, overlap_h);

    // Determine the appropriate SSIM gate before printing the summary.
    // Below 1280 px the stackr-vs-zerene comparison is cross-resolution
    // (stackr sourced from low-res inputs, zerene from full 3840px source),
    // which structurally depresses SSIM regardless of alignment quality.
    // SSIM_MIN_LOWRES (0.30) still rejects badly ghosted/misaligned results
    // while passing clean stacks that are merely resolution-limited.
    let ssim_gate = if inputs_comparable_res {
        SSIM_MIN
    } else {
        SSIM_MIN_LOWRES
    };
    let ssim_gate_label = if inputs_comparable_res {
        "hi-res"
    } else {
        "lo-res"
    };

    eprintln!("[quality test] SSIM(stackr, zerene) after shift alignment = {quality_ssim:.4}");
    eprintln!("[quality test] (overlap region: {overlap_w}×{overlap_h} px at SSIM scale)");
    eprintln!("[quality test] --- THRESHOLD SUMMARY ---");
    eprintln!(
        "[quality test]   sharpness ratio:    {sharp_ratio:.4} (threshold >= {SHARPNESS_MIN_RATIO})"
    );
    eprintln!(
        "[quality test]   SSIM:               {quality_ssim:.4} (threshold >= {ssim_gate} [{ssim_gate_label}])"
    );
    eprintln!(
        "[quality test]   stackr vs best-1fr: {:.4} (threshold >= {BEATS_SINGLE_FRAME})",
        t_stackr / best_single_t.max(1e-9)
    );

    // Write metrics under the build directory (never into the source tree)
    // so they are readable after the run even when the harness captures
    // per-thread stderr.  One file per variant.
    let metrics_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/quality");
    std::fs::create_dir_all(&metrics_dir).expect("create target/quality");
    let metrics_path = metrics_dir.join(format!("quality_metrics_{variant_tag}.txt"));
    let metrics_txt = format!(
        "tenengrad_stackr={t_stackr:.6}\ntenengrad_zerene={t_zerene:.6}\nsharp_ratio={sharp_ratio:.6}\nssim={quality_ssim:.6}\nssim_gate={ssim_gate}\nssim_gate_label={ssim_gate_label}\nbest_single_t={best_single_t:.6}\ncoarse_dx={dx}\ncoarse_dy={dy}\n"
    );
    let _ = std::fs::write(&metrics_path, &metrics_txt);
    eprintln!(
        "[quality test] metrics written to {}",
        metrics_path.display()
    );

    assert!(
        quality_ssim >= ssim_gate,
        "SSIM(stackr, zerene)={quality_ssim:.4} < {ssim_gate:.2} ({ssim_gate_label} gate); possible ghosting or misalignment"
    );

    // Cleanup.
    let _ = std::fs::remove_dir_all(&out_dir);
    if let Some(ref tmp) = tmp_cleanup {
        let _ = std::fs::remove_dir_all(tmp);
    }
}

/// **Tiled `Relief` equals whole-image `Relief`** (TST-2, FUS-4).
///
/// 96×96 pixels, 32-px tiles (3×3), the `Relief` apron derived from the radii
/// (5 + 2·2 + 1 = 10 px): every padded tile is a strict part of the image. The
/// contrast threshold is computed **once** from the whole-image focus map and
/// applied to every tile, exactly as the CLI's fusion pass does; evaluating the
/// percentile (or the median + 1.5·MAD auto-detect) per tile would change the mask from tile to tile.
#[test]
fn test_tiled_matches_whole_image_relief() {
    const TOL: f32 = 0.01; // the guided filter widens the neighbourhood: slightly looser

    let frames = multiscale_stack(96, 96, 3);
    for (contrast_pct, auto_detect) in [(0.3_f32, false), (0.0, true)] {
        let settings = StackingSettings {
            relief_contrast_pct: contrast_pct,
            relief_auto_detect: auto_detect,
            ..StackingSettings::default()
        };
        let geometry = Geometry::choose(32, Method::Relief, &settings);
        assert_eq!((geometry.tile_size, geometry.apron), (32, 10));

        let whole = run_relief(&frames, &settings);
        let maps = focus_maps(&frames, settings.relief_estimation_radius as usize);
        let threshold = relief_threshold(&maps.max.luma, &settings);
        let tiled = fuse_in_tiles(&frames, geometry, |t| {
            relief_tile(t, &settings, threshold.value)
        });

        let [luma, ..] = max_plane_error(&tiled, &whole);
        assert!(
            luma <= TOL,
            "Relief (pct {contrast_pct}, auto {auto_detect}) tiled luma deviates by {luma:.6} \
             (tol={TOL})"
        );
    }
}

// ── The scratch directory (the shared `tiled::ScratchDir`) ────────────────────

#[test]
fn scratch_directory_exists_while_guarded_and_is_gone_after_drop() {
    let guard = ScratchDir::create(&std::env::temp_dir()).expect("create");
    let path = guard.path().to_path_buf();
    std::fs::write(path.join("tile.bin"), b"data").expect("write");
    assert!(path.is_dir());
    drop(guard);
    assert!(!path.exists());
}

#[test]
fn scratch_directory_is_removed_when_a_panic_unwinds() {
    let root = std::env::temp_dir();
    let seen = std::sync::Mutex::new(PathBuf::new());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let guard = ScratchDir::create(&root).expect("create");
        *seen.lock().unwrap() = guard.path().to_path_buf();
        panic!("forced failure inside the guarded scope");
    }));
    assert!(result.is_err());
    let path = seen.lock().unwrap().clone();
    assert!(!path.as_os_str().is_empty());
    assert!(!path.exists(), "guard must clean up on unwind");
}

// ── Output names of a run with several algorithms ─────────────────────────────

#[test]
fn suffixed_path_keeps_directory_and_extension() {
    assert_eq!(
        suffixed_path(Path::new("out/result.png"), "apex"),
        Some(PathBuf::from("out/result_apex.png"))
    );
    assert_eq!(
        suffixed_path(Path::new("result"), "relief"),
        Some(PathBuf::from("result_relief"))
    );
    assert_eq!(suffixed_path(Path::new(""), "apex"), None);
}
