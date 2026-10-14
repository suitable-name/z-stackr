//! The device seam, progress and preview callbacks, and streamed frame sources
//! of the Strata fusion.
#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

use stacker_algo::{
    relief::guided::guided_filter_pair_luma,
    strata::{
        CpuStrata, FnFrames, NO_PREVIEW, StrataKernels, StrataParams, fuse_strata, fuse_strata_on,
        fuse_strata_with_preview, fuse_strata_with_progress, saliency::compute_saliency,
    },
};
use stacker_core::image::PlanarImage;
use std::sync::atomic::{AtomicUsize, Ordering};

fn frame(w: usize, h: usize, phase: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            // A stripe pattern whose sharp band moves with the phase, so the
            // winner differs between frames.
            let band = ((y as f32 / h as f32) * 3.0 - phase).abs() < 0.5;
            let v = if band {
                if (x + y) % 2 == 0 { 0.8 } else { 0.2 }
            } else {
                0.5 + 0.02 * ((x as f32) * 0.1).sin()
            };
            img.luma[i] = v;
            img.chroma_a[i] = (v - 0.5) * 0.2;
            img.chroma_b[i] = (0.5 - v) * 0.1;
        }
    }
    img
}

fn stack() -> Vec<PlanarImage<f32>> {
    (0..3).map(|k| frame(40, 36, k as f32)).collect()
}

const fn params() -> StrataParams {
    StrataParams {
        base_radius: 8,
        detail_focus: 3,
    }
}

fn assert_bit_equal(a: &PlanarImage<f32>, b: &PlanarImage<f32>) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    for (pa, pb) in [
        (&a.luma, &b.luma),
        (&a.chroma_a, &b.chroma_a),
        (&a.chroma_b, &b.chroma_b),
    ] {
        assert!(
            pa.iter().zip(pb).all(|(x, y)| x.to_bits() == y.to_bits()),
            "planes differ bitwise"
        );
    }
}

/// Serves both kernels with the very functions the CPU default uses.
struct Explicit {
    saliency_calls: usize,
    pair_calls: usize,
}

impl StrataKernels for Explicit {
    fn saliency(&mut self, luma: &[f32], width: usize, height: usize) -> Option<Vec<f32>> {
        self.saliency_calls += 1;
        Some(compute_saliency(luma, width, height))
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
        self.pair_calls += 1;
        Some(guided_filter_pair_luma(
            guidance, src, width, height, radius_a, eps_a, radius_b, eps_b,
        ))
    }
}

/// Answers with results of the wrong length; the CPU must run instead.
struct Broken;

impl StrataKernels for Broken {
    fn saliency(&mut self, _: &[f32], _: usize, _: usize) -> Option<Vec<f32>> {
        Some(vec![1.0; 3])
    }

    fn guided_pair(
        &mut self,
        _: &[f32],
        _: &[f32],
        _: usize,
        _: usize,
        _: usize,
        _: f32,
        _: usize,
        _: f32,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        Some((vec![0.0; 2], vec![0.0; 2]))
    }
}

#[test]
fn cpu_kernels_through_the_seam_equal_fuse_strata() {
    let frames = stack();
    let expected = fuse_strata(&frames, &params());
    let via_seam = fuse_strata_on(
        &mut CpuStrata,
        frames.as_slice(),
        &params(),
        || {},
        NO_PREVIEW,
        |_: &PlanarImage<f32>| {},
    );
    assert_bit_equal(&expected, &via_seam);
}

#[test]
fn explicit_kernels_equal_the_default_and_are_called_once_per_frame() {
    let frames = stack();
    let expected = fuse_strata(&frames, &params());
    let mut kernels = Explicit {
        saliency_calls: 0,
        pair_calls: 0,
    };
    let fused = fuse_strata_on(
        &mut kernels,
        frames.as_slice(),
        &params(),
        || {},
        NO_PREVIEW,
        |_: &PlanarImage<f32>| {},
    );
    assert_bit_equal(&expected, &fused);
    assert_eq!(kernels.saliency_calls, frames.len());
    assert_eq!(kernels.pair_calls, frames.len());
}

#[test]
fn kernel_results_of_the_wrong_length_fall_back_to_the_cpu() {
    let frames = stack();
    let expected = fuse_strata(&frames, &params());
    let fused = fuse_strata_on(
        &mut Broken,
        frames.as_slice(),
        &params(),
        || {},
        NO_PREVIEW,
        |_: &PlanarImage<f32>| {},
    );
    assert_bit_equal(&expected, &fused);
}

#[test]
fn progress_ticks_twice_per_frame() {
    let frames = stack();
    let mut ticks = 0_usize;
    let fused = fuse_strata_with_progress(&frames, &params(), || ticks += 1);
    assert_eq!(ticks, 2 * frames.len());
    assert_bit_equal(&fused, &fuse_strata(&frames, &params()));
}

#[test]
fn previews_follow_the_cadence_and_end_with_the_result() {
    let frames: Vec<_> = (0..5).map(|k| frame(40, 36, k as f32 * 0.5)).collect();
    let mut previews = Vec::new();
    let fused = fuse_strata_with_preview(
        &frames,
        &params(),
        || {},
        2,
        |p| {
            previews.push(p.clone());
        },
    );
    // After frames 2 and 4, then once more with the finished image.
    assert_eq!(previews.len(), 3);
    assert_bit_equal(previews.last().expect("a final preview"), &fused);
    assert_bit_equal(&fused, &fuse_strata(&frames, &params()));
}

#[test]
fn a_preview_cadence_of_zero_means_every_frame() {
    let frames = stack();
    let mut count = 0_usize;
    let _ = fuse_strata_with_preview(&frames, &params(), || {}, 0, |_| count += 1);
    // After frames 1 and 2, then the final one.
    assert_eq!(count, frames.len());
}

#[test]
fn no_preview_switches_the_callback_off() {
    let frames = stack();
    let mut count = 0_usize;
    let _ = fuse_strata_with_preview(&frames, &params(), || {}, NO_PREVIEW, |_| count += 1);
    assert_eq!(count, 0);
}

#[test]
fn a_streamed_source_gives_the_same_result() {
    let frames = stack();
    let loads = AtomicUsize::new(0);
    let source = FnFrames::new(frames.len(), |i| {
        loads.fetch_add(1, Ordering::Relaxed);
        frames[i].clone()
    });
    let streamed = fuse_strata_on(
        &mut CpuStrata,
        &source,
        &params(),
        || {},
        NO_PREVIEW,
        |_: &PlanarImage<f32>| {},
    );
    assert_bit_equal(&streamed, &fuse_strata(&frames, &params()));
    // Two passes over the frames (plus the size probe of the first one).
    assert!(loads.load(Ordering::Relaxed) >= 2 * frames.len());
}
