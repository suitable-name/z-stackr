//! Timing harness — not a correctness test, but printed as a timing report.
//!
//! Run with:
//!   cargo test -p stacker-algo `apex_parallel_speedup` -- --nocapture --ignored
//!
//! The test is marked `#[ignore]` so it does not run in CI (it is slow and
//! has no assertion that could fail); pass `--ignored` to opt in.

use stacker_algo::apex::{
    fuse::{build_and_fuse_pyramids, fuse_pyramids},
    pyramid::LaplacianPyramid,
};
use stacker_core::image::PlanarImage;
use std::time::Instant;

const W: usize = 512;
const H: usize = 512;
const N: usize = 8;
const LEVELS: usize = 6;
const REPS: u32 = 3;

fn make_frame(phase: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let v = ((x as f32).mul_add(0.05, phase).sin()
                * phase.mul_add(1.3, y as f32 * 0.04).cos())
            .mul_add(0.5, 0.5)
            .clamp(0.0, 1.0);
            img.luma[i] = v;
            img.chroma_a[i] = (v - 0.5) * 0.3;
            img.chroma_b[i] = (0.5 - v) * 0.2;
        }
    }
    img
}

#[test]
#[ignore = "slow timing harness: run with --ignored --nocapture to see speedup numbers"]
fn apex_parallel_speedup() {
    let frames: Vec<PlanarImage<f32>> = (0..N)
        .map(|k| make_frame(k as f32 * std::f32::consts::PI / N as f32))
        .collect();

    // ── Serial: build pyramids one-by-one then fuse ──────────────────────
    let mut serial_total = std::time::Duration::ZERO;
    for _ in 0..REPS {
        let t = Instant::now();
        let pyramids: Vec<LaplacianPyramid> = frames
            .iter()
            .map(|img| LaplacianPyramid::build(img, LEVELS))
            .collect();
        let fused = fuse_pyramids(&pyramids, false, true);
        let _ = fused.reconstruct();
        serial_total += t.elapsed();
    }
    let serial_ms = serial_total.as_secs_f64() * 1000.0 / f64::from(REPS);

    // ── Parallel: par_iter across images ─────────────────────────────────
    let mut par_total = std::time::Duration::ZERO;
    for _ in 0..REPS {
        let t = Instant::now();
        let fused = build_and_fuse_pyramids(&frames, LEVELS, false, true);
        let _ = fused.reconstruct();
        par_total += t.elapsed();
    }
    let par_ms = par_total.as_secs_f64() * 1000.0 / f64::from(REPS);

    let speedup = serial_ms / par_ms;
    println!(
        "\n[apex timing] {N} frames × {W}×{H}, {LEVELS} levels, {REPS} reps\n\
         serial  : {serial_ms:.1} ms/iter\n\
         parallel: {par_ms:.1} ms/iter\n\
         speedup : {speedup:.2}×\n"
    );

    // Sanity: parallel must not be more than 2× slower than serial
    // (a regression guard, not a performance target).
    assert!(
        par_ms < serial_ms * 2.0,
        "parallel path ({par_ms:.1} ms) is more than 2× slower than serial ({serial_ms:.1} ms) — likely a regression"
    );
}
