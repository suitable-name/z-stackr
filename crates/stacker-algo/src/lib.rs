//! The fusion algorithms: how a stack of aligned frames becomes one sharp image.
//!
//! * [`apex`]: Laplacian-pyramid fusion. Each pyramid band takes, per pixel, the
//!   coefficient of the frame with the largest energy; the base is averaged.
//! * [`relief`]: depth-map fusion. A plane-fit residual-variance focus measure
//!   picks the sharpest frame per pixel, a threshold masks flat areas, and a
//!   guided filter smooths the result along its own edges.
//! * [`strata`]: guided-filter soft blending. Every frame contributes to every
//!   pixel, weighted by its local sharpness; the weights are refined along the
//!   image's edges at two scales (base and detail layer).
//! * [`hybrid`]: the retouch session: a brush paints an alpha mask that blends a
//!   second image (another frame or fusion result) over the fused base, with a
//!   block-diff undo history.
//! * [`optimize`]: the auto-cull, which ranks frames by the in-focus detail
//!   they win, drops the ones that add nothing and orders the rest.
//!
//! Ties are broken the same way everywhere: the frame that comes first in the
//! input order wins. No function here draws random numbers, and every parallel
//! reduction either writes disjoint output rows or combines exactly
//! representable values (maxima, integer counts), so results do not depend on the
//! thread count.

#![feature(portable_simd)]
#![warn(missing_docs)]

/// Laplacian-pyramid fusion.
pub mod apex {
    /// Per-band selection of the strongest coefficient and the fusion entry
    /// points.
    pub mod fuse;
    /// The pyramid blur, decimation, expansion and the pyramid type.
    pub mod pyramid;
}
/// Depth-map fusion: focus measure, threshold, guided smoothing and the
/// multigrid depth solver.
pub mod relief {
    /// The plane-fit residual-variance focus measure.
    pub mod focus;
    /// The guided filter.
    pub mod guided;
    /// The multigrid depth solver and the depth compositing.
    pub mod multigrid;
    /// Gaussian-pyramid smoothing of the solved depth.
    pub mod pyramid;
    /// Contrast thresholds for the focus measure (percentile and median + MAD).
    pub mod threshold;
}
pub mod strata;
/// Retouching of fused results.
pub mod hybrid {
    /// The retouch session and its undo history.
    pub mod retouch;
}
/// The auto-cull and fusion-order recommendation.
pub mod optimize;

#[cfg(test)]
mod timing {
    //! Timing harness — a report only; nothing about elapsed time is asserted.
    //!
    //! Run with:
    //!   `cargo test -p z-stackr-algo apex_parallel_speedup -- --nocapture --ignored`
    //!
    //! The test is marked `#[ignore]` so it does not run by default (it is slow
    //! and depends on the machine's load); pass `--ignored` to opt in.  It
    //! prints the serial and parallel times and their ratio, and asserts only
    //! that both paths return the same image.

    use crate::apex::{
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

    #[expect(
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        reason = "pixel indices are far below 2^24; the synthetic pattern is spelled out as plain multiply-add"
    )]
    fn make_frame(phase: f32) -> PlanarImage<f32> {
        let mut img = PlanarImage::new(W, H);
        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                let v =
                    ((x as f32 * 0.05 + phase).sin() * (y as f32 * 0.04 + phase * 1.3).cos() * 0.5
                        + 0.5)
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
    #[expect(clippy::cast_precision_loss, reason = "frame counts are tiny")]
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

        // Report-only: no assertion about wall-clock time, which depends on
        // machine load and the rayon thread count.  The only property checked
        // is that the two paths produce the same image.
        let serial_pyramids: Vec<LaplacianPyramid> = frames
            .iter()
            .map(|img| LaplacianPyramid::build(img, LEVELS))
            .collect();
        let serial_image = fuse_pyramids(&serial_pyramids, false, true).reconstruct();
        let parallel_image = build_and_fuse_pyramids(&frames, LEVELS, false, true).reconstruct();
        assert_eq!(
            serial_image.luma, parallel_image.luma,
            "serial and parallel Apex must agree"
        );
    }
}
