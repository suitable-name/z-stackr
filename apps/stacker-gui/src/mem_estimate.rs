//! Rough peak-memory estimation for the GUI's default in-RAM (non-tiled)
//! stacking pipeline (`StackingSettings::tile_size == 0`), plus a look at how
//! much system RAM is actually available.
//!
//! By default the GUI decodes and aligns every frame into RAM up front and
//! keeps it all resident at once. That's deliberate, not an oversight: the
//! live per-frame alignment preview and the "Auto-Cull Frames" pass both
//! need to compare every frame against every other frame, which only works
//! with the whole stack in memory simultaneously (see the "Auto-Cull Frames"
//! note in the project README for the same trade-off spelled out for that
//! feature).
//!
//! Setting `StackingSettings::tile_size > 0` switches the Stack handler to
//! the shared out-of-core `stacker_pipeline::run_pipeline` engine instead
//! (the same one the CLI's `--tile-size` always uses) — see
//! `main.rs::on_request_stack` — whose peak memory scales with tile area
//! rather than stack size, at the cost of Auto-Cull and full live preview.
//! This module's estimate only applies to the default `tile_size == 0` path;
//! it's what tells the user up front roughly how much RAM *that* path will
//! need, so they can back off (fewer frames, a resize-percent preprocessing
//! step, or switch on tiling) before hitting a slow swap-thrashing run or an
//! OOM.

use std::mem::size_of;

use sysinfo::System;

/// Bytes per pixel for one decoded+aligned frame: `PlanarImage<f32>`
/// (`stacker_core::image`) stores three full-resolution `f32` channels —
/// luma, `chroma_a`, `chroma_b`.
const BYTES_PER_PIXEL_PER_FRAME: u64 = 3 * size_of::<f32>() as u64;

/// Conservative multiplier for the extra whole-stack copies the GUI
/// pipeline can hold at once beyond the base decoded/aligned buffer:
/// * the `AlignedCache` (a full clone, written after every Align/Stack run
///   so a later run can skip re-aligning unchanged input);
/// * the auto-cull re-ordered copy (when culling actually reorders frames);
/// * Apex/Relief's own per-frame working buffers (Laplacian pyramids, focus
///   maps).
///
/// This is deliberately an order-of-magnitude estimate, not an exact
/// accounting — the goal is a useful warning, not a byte-perfect figure.
const PEAK_MULTIPLIER: f64 = 3.0;

/// Estimates peak RAM (in bytes) the GUI's in-RAM pipeline will use to
/// align and fuse `frame_count` frames of `width` x `height` pixels.
#[must_use]
pub fn estimate_peak_bytes(frame_count: usize, width: u32, height: u32, akaze: bool) -> u64 {
    let per_frame_bytes = u64::from(width) * u64::from(height) * BYTES_PER_PIXEL_PER_FRAME;
    let base_bytes = per_frame_bytes as f64 * frame_count as f64;
    let mut peak_bytes = (base_bytes * PEAK_MULTIPLIER) as u64;

    if akaze {
        // AKAZE builds an internal non-linear scale space (evolution) and extracts
        // features. This requires several full-resolution f32 buffers (L, Lsmooth,
        // Lx, Ly, Lxx, Lxy, Lyy, Lt, etc.).
        // A single frame's AKAZE extraction can take roughly 20x the frame size in RAM.
        // Because coarse seed matching runs in parallel via Rayon (`into_par_iter`),
        // up to `num_cpus` threads may be allocating these heavy buffers at once.
        let cpus = std::thread::available_parallelism().map_or(1, std::num::NonZero::get) as u64;
        let concurrent_akaze_tasks = cpus.min(frame_count as u64);

        let akaze_buffers_per_task = (u64::from(width) * u64::from(height)) * 4 * 20;
        peak_bytes += concurrent_akaze_tasks * akaze_buffers_per_task;
    }

    peak_bytes
}

/// Total system RAM in bytes, if it could be determined for this platform.
#[must_use]
pub fn total_system_memory_bytes() -> Option<u64> {
    let mut sys = System::new();
    sys.refresh_memory();
    let total = sys.total_memory();
    (total > 0).then_some(total)
}

/// Formats a byte count as a short human-readable size, e.g. `"4.2 GB"`.
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.1} GB", b / GIB)
    } else {
        format!("{:.0} MB", (b / MIB).max(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::{estimate_peak_bytes, format_bytes};

    #[test]
    fn estimate_scales_with_frame_count_and_resolution() {
        let one_frame = estimate_peak_bytes(1, 1000, 1000, false);
        let ten_frames = estimate_peak_bytes(10, 1000, 1000, false);
        assert_eq!(ten_frames, one_frame * 10);

        let bigger = estimate_peak_bytes(1, 2000, 2000, false);
        assert_eq!(bigger, one_frame * 4);
    }

    #[test]
    fn estimate_is_zero_for_zero_frames() {
        assert_eq!(estimate_peak_bytes(0, 4000, 3000, false), 0);
    }

    #[test]
    fn format_bytes_picks_a_sensible_unit() {
        assert_eq!(format_bytes(500 * 1024 * 1024), "500 MB");
        assert_eq!(
            format_bytes(4 * 1024 * 1024 * 1024 + 512 * 1024 * 1024),
            "4.5 GB"
        );
    }
}
