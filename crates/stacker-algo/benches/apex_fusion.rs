//! Criterion benchmark for the `Apex` fusion pipeline.
//!
//! Two groups are measured:
//! - `apex_fusion/parallel` — `build_and_fuse_pyramids` (rayon `par_iter` across images)
//! - `apex_fusion/serial`   — equivalent serial loop (baseline for speedup calculation)
//!
//! Run: `cargo bench -p stacker-algo --bench apex_fusion`
#![allow(
    clippy::cast_precision_loss, // synthetic benchmark: usize→f32 for pixel coords is fine
    clippy::suboptimal_flops,    // readability preferred in benchmark setup code
)]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use stacker_algo::apex::{
    fuse::{build_and_fuse_pyramids, fuse_pyramids},
    pyramid::LaplacianPyramid,
};
use stacker_core::image::PlanarImage;

/// Build a synthetic `PlanarImage<f32>` with a sinusoidal pattern.
fn make_frame(w: usize, h: usize, phase: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let v = ((x as f32 * 0.05 + phase).sin() * (y as f32 * 0.04 + phase * 1.3).cos() * 0.5
                + 0.5)
                .clamp(0.0, 1.0);
            img.luma[i] = v;
            img.chroma_a[i] = (v - 0.5) * 0.3;
            img.chroma_b[i] = (0.5 - v) * 0.2;
        }
    }
    img
}

fn bench_apex_fusion(c: &mut Criterion) {
    const W: usize = 512;
    const H: usize = 512;
    const N_FRAMES: usize = 8;
    const MAX_LEVELS: usize = 6;

    let frames: Vec<PlanarImage<f32>> = (0..N_FRAMES)
        .map(|k| make_frame(W, H, k as f32 * std::f32::consts::PI / N_FRAMES as f32))
        .collect();

    // Each iteration is heavy; 10 samples gives stable numbers without
    // running for many minutes.
    let mut group = c.benchmark_group("apex_fusion");
    group.sample_size(10);

    // ── Parallel baseline (rayon par_iter across images) ─────────────────
    group.bench_with_input(
        BenchmarkId::new("parallel", format!("{N_FRAMES}x{W}x{H}")),
        &frames,
        |b, frames| {
            b.iter(|| {
                let fused = build_and_fuse_pyramids(frames, MAX_LEVELS, false, true);
                let _out = fused.reconstruct();
            });
        },
    );

    // ── Serial baseline (iter across images, no rayon) ────────────────────
    group.bench_with_input(
        BenchmarkId::new("serial", format!("{N_FRAMES}x{W}x{H}")),
        &frames,
        |b, frames| {
            b.iter(|| {
                let pyramids: Vec<LaplacianPyramid> = frames
                    .iter()
                    .map(|img| LaplacianPyramid::build(img, MAX_LEVELS))
                    .collect();
                let fused = fuse_pyramids(&pyramids, false, true);
                let _out = fused.reconstruct();
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_apex_fusion);
criterion_main!(benches);
