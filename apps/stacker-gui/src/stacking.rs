use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::{Arc, Mutex},
};

use slint::{Rgba8Pixel, SharedPixelBuffer};

use stacker_algo::{
    apex::fuse::fuse_pyramids_incremental_with_progress,
    relief::threshold::{ReliefSettings, generate_mask},
    strata::{StrataParams, fuse_strata_with_preview},
};
use stacker_core::image::PlanarImage;

use crate::settings::StackingSettings;

// ── Real fusion entry points ──────────────────────────────────────────────────

/// `Apex` fusion: Laplacian pyramid multi-resolution maximum contrast.
///
/// `on_progress(completed, total, preview)` is forwarded from
/// [`fuse_pyramids_incremental_with_progress`]: it fires once per frame so
/// the caller can report live progress, and carries a freshly reconstructed
/// preview image on a throttled subset of frames so the caller can refresh
/// the displayed image while fusion is still running instead of the GUI
/// appearing to hang until the whole stack is done.
pub fn fuse_apex<F>(
    planar_imgs: &[PlanarImage<f32>],
    settings: &StackingSettings,
    on_progress: F,
) -> PlanarImage<f32>
where
    F: FnMut(usize, usize, Option<PlanarImage<f32>>),
{
    fuse_pyramids_incremental_with_progress(
        planar_imgs,
        settings.use_all_color_channels,
        settings.grit_suppression,
        on_progress,
    )
}

/// `Strata` fusion: guided-filter soft-blend (see `docs/strata-fusion-design.md`).
///
/// `on_progress(completed, total)` fires once per frame per internal pass
/// (`2 * planar_imgs.len()` ticks total), mirroring `fuse_apex`'s per-frame
/// live-progress reporting.
pub fn fuse_strata<F>(
    planar_imgs: &[PlanarImage<f32>],
    settings: &StackingSettings,
    on_progress: F,
) -> PlanarImage<f32>
where
    F: FnMut(usize, usize, Option<PlanarImage<f32>>),
{
    let params = StrataParams {
        base_radius: settings.strata_base_radius as usize,
        detail_focus: settings.strata_detail_focus,
    };
    let total = 2 * planar_imgs.len();
    let on_progress_cell = Rc::new(RefCell::new(on_progress));
    let completed_count = Rc::new(Cell::new(0usize));

    let on_progress_cell_1 = on_progress_cell.clone();
    let completed_count_1 = completed_count.clone();

    fuse_strata_with_preview(
        planar_imgs,
        &params,
        &mut |completed| {
            completed_count_1.set(completed);
            (*on_progress_cell_1.borrow_mut())(completed, total, None);
        },
        1, // preview_every: 1 frames is fine for Strata's GPU pass
        |preview| {
            (*on_progress_cell.borrow_mut())(completed_count.get(), total, Some(preview.clone()));
        },
    )
}

/// `Relief` fusion: `SML` per-pixel source selection + guided-filter luma smoothing.
///
/// # Contrast mask invariant
/// When `settings.relief_contrast_pct == 0.0` (the default), `generate_mask`
/// selects the 0th-percentile threshold (the minimum SML value), so every
/// pixel is ≥ that threshold and every mask entry is `true`.  The fallback
/// branch is therefore never taken, and the output is **byte-identical** to
/// the pre-mask pipeline.
///
/// # Panics
///
/// Calls `.lock().unwrap()` on the `Mutex`-wrapped `slint::Weak` handle
/// passed to the focus-measure progress callback, and on `preview_ctx_arc`;
/// those panic only if another thread already panicked while holding the
/// same lock (mutex poisoning), which does not happen in normal operation.
// `too_many_lines`: one cohesive Relief fusion pipeline (focus-measure
// progress, optional interactive contrast-preview round-trip, depth-map
// preview, multigrid/mask solve dispatch) — the interactive preview's
// channel round-trip (`tx`/`rx`) ties several of these steps' state
// together, so splitting it would scatter that shared state across
// artificial function boundaries.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn fuse_relief(
    planar_imgs: &[PlanarImage<f32>],
    settings: &StackingSettings,
    base_image: Option<&image::RgbaImage>,
    app_weak: Option<&slint::Weak<crate::App>>,
    preview_ctx_arc: Option<&Arc<Mutex<Option<ReliefPreviewContext>>>>,
) -> Option<PlanarImage<f32>> {
    if planar_imgs.is_empty() {
        return Some(PlanarImage::new(0, 0));
    }

    // ── Live progress: per-frame focus-measure computation ──────────────────
    // Mirrors the Apex fusion loop's per-frame reporting: the SML stage is by
    // far the longest Relief phase and scales linearly with the frame count, so
    // per-frame updates give the user real progress instead of a stalled bar.
    // Total-progress mapping: the fusion stage owns 0.50..0.95 of the overall
    // bar (alignment ends at 0.50 — see main.rs); the SML stage takes the
    // first part of that span.
    let n_frames = planar_imgs.len();
    // `slint::Weak` is `Send` but not `Sync`; the progress callback below is
    // invoked from rayon worker threads through a shared `&` closure, so the
    // weak handle is parked behind a `Mutex` (making the capture `Sync`) and
    // cloned per call for the `invoke_from_event_loop` hop.
    let weak_for_workers = app_weak.map(|w| std::sync::Mutex::new(w.clone()));
    let (smls, max_sml) = stacker_algo::relief::fuse::compute_relief_smls_with_progress(
        planar_imgs,
        settings.relief_estimation_radius as usize,
        |completed, total| {
            let Some(m) = &weak_for_workers else { return };
            let weak = m.lock().unwrap().clone();
            let frac = completed as f32 / total as f32;
            report_relief_progress(
                Some(&weak),
                format!("Fusing [Relief] — focus measure {completed}/{total}…"),
                0.50 + 0.20 * frac,
                frac,
            );
        },
    );

    let mut final_contrast_pct = settings.relief_contrast_pct;

    if settings.relief_show_preview
        && let (Some(base_img), Some(app_weak), Some(ctx_arc)) =
            (base_image, app_weak, preview_ctx_arc)
    {
        let state = ReliefPreviewState {
            max_sml: max_sml.clone(),
            base_image: base_img.clone(),
            est_radius: settings.relief_estimation_radius as usize,
            smooth_radius: settings.relief_smoothing_radius as usize,
        };

        let (tx, rx) = std::sync::mpsc::channel::<Option<f32>>();

        *ctx_arc.lock().unwrap() = Some(ReliefPreviewContext {
            state: state.clone(),
            reply_tx: tx,
        });

        if settings.relief_auto_detect {
            final_contrast_pct = stacker_algo::relief::fuse::auto_contrast_threshold(&max_sml);
        }

        let initial_contrast = final_contrast_pct;
        let initial_buf = generate_relief_preview(&state, initial_contrast);
        let _ = slint::invoke_from_event_loop({
            let a = app_weak.clone();
            move || {
                if let Some(app) = a.upgrade() {
                    app.set_set_relief_contrast_pct(initial_contrast);
                    app.set_relief_preview_image(slint::Image::from_rgba8(initial_buf));
                    app.set_relief_preview_open(true);
                }
            }
        });

        if let Ok(Some(new_contrast)) = rx.recv() {
            final_contrast_pct = new_contrast;
        } else {
            *ctx_arc.lock().unwrap() = None;
            return None;
        }

        *ctx_arc.lock().unwrap() = None;
    } else if settings.relief_auto_detect {
        final_contrast_pct = stacker_algo::relief::fuse::auto_contrast_threshold(&max_sml);
    }

    let relief_settings = ReliefSettings {
        est_radius: settings.relief_estimation_radius as usize,
        smooth_radius: settings.relief_smoothing_radius as usize,
        contrast_pct: final_contrast_pct,
        // GUI in-RAM path fuses the whole image in one shot (no tiling), so
        // there is no cross-tile threshold-consistency problem to solve here
        // — keep deriving the threshold from contrast_pct/otsu as before.
        absolute_threshold: None,
    };

    // ── Live preview: show the raw depth-index map before the solve ─────────
    // Relief has no per-frame incremental composite the way Apex does (the
    // solve/selection is a single whole-image pass), but the per-pixel argmax
    // of the focus measures IS the raw depth map the solver starts from — a
    // meaningful, cheap-to-render intermediate. Displaying it gives the same
    // "something is visibly happening" feedback the Apex path provides.
    if let Some(app_weak) = app_weak
        && n_frames > 1
    {
        let preview = depth_index_preview(&smls);
        let buf = crate::image_utils::planar_to_rgba_buffer(&preview);
        let _ = slint::invoke_from_event_loop({
            let a = app_weak.clone();
            move || {
                if let Some(app) = a.upgrade() {
                    app.set_displayed_image(slint::Image::from_rgba8(buf));
                }
            }
        });
    }

    let result = if settings.relief_use_multigrid {
        report_relief_progress(
            app_weak,
            "Fusing [Relief] — solving depth map (multigrid)…".to_owned(),
            0.80,
            0.80,
        );
        stacker_algo::relief::fuse::fuse_relief_multigrid(
            planar_imgs,
            &smls,
            &max_sml,
            &relief_settings,
        )
    } else {
        report_relief_progress(
            app_weak,
            "Fusing [Relief] — selecting sharpest pixels…".to_owned(),
            0.80,
            0.80,
        );
        stacker_algo::relief::fuse::fuse_relief_with_mask(
            planar_imgs,
            &smls,
            &max_sml,
            &relief_settings,
        )
    };
    report_relief_progress(app_weak, "Fusing [Relief] — done.".to_owned(), 0.95, 1.0);
    Some(result)
}

/// Grayscale preview of the per-pixel argmax depth index across the SML maps
/// (0 = first frame = dark, last frame = bright). This is the raw depth field
/// the `Relief` solvers refine, rendered as a quick visual progress intermediate.
fn depth_index_preview(smls: &[PlanarImage<f32>]) -> PlanarImage<f32> {
    use rayon::prelude::*;

    let width = smls[0].width;
    let height = smls[0].height;
    let len = width * height;

    let max_idx = (smls.len().saturating_sub(1)).max(1) as f32;

    let mut luma = vec![0.0_f32; len];
    luma.par_iter_mut().enumerate().for_each(|(i, out)| {
        let mut best = f32::NEG_INFINITY;
        let mut best_idx = 0usize;
        for (idx, sml) in smls.iter().enumerate() {
            if sml.luma[i] > best {
                best = sml.luma[i];
                best_idx = idx;
            }
        }

        {
            *out = 0.05 + 0.9 * (best_idx as f32 / max_idx);
        }
    });

    PlanarImage {
        width,
        height,
        luma,
        chroma_a: vec![0.0; len],
        chroma_b: vec![0.0; len],
    }
}

/// Post an interim status + progress update to the GUI during a `Relief` fusion
/// phase. `total` drives the overall progress bar (monotonic across the whole
/// stack run: fusion owns the 0.50..0.95 span), `step` drives the per-step
/// bar (0..1 within the `Relief` stage). A no-op when `app_weak` is `None` (e.g.
/// when `fuse_relief` is called from a context with no UI to report to).
fn report_relief_progress(
    app_weak: Option<&slint::Weak<crate::App>>,
    status: String,
    total: f32,
    step: f32,
) {
    let Some(app_weak) = app_weak else { return };
    let _ = slint::invoke_from_event_loop({
        let a = app_weak.clone();
        move || {
            if let Some(app) = a.upgrade() {
                app.set_status(status.into());
                app.set_progress(total);
                app.set_step_progress(step);
            }
        }
    });
}

// ── Relief preview types and helpers ────────────────────────────────────────────

#[derive(Clone)]
pub struct ReliefPreviewState {
    pub max_sml: PlanarImage<f32>,
    pub base_image: image::RgbaImage,
    pub est_radius: usize,
    pub smooth_radius: usize,
}

#[must_use]
pub fn generate_relief_preview(
    state: &ReliefPreviewState,
    contrast_pct: f32,
) -> SharedPixelBuffer<Rgba8Pixel> {
    let relief_settings = ReliefSettings {
        est_radius: state.est_radius,
        smooth_radius: state.smooth_radius,
        contrast_pct,
        absolute_threshold: None,
    };
    let mask = generate_mask(&state.max_sml, &relief_settings);

    let w = state.base_image.width();
    let h = state.base_image.height();
    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(w, h);

    for (i, p) in state.base_image.pixels().enumerate() {
        let mut r = p[0];
        let mut g = p[1];
        let mut b = p[2];
        if !mask.get(i).copied().unwrap_or(true) {
            r = r.saturating_add(100);
            g = g.saturating_sub(50);
            b = b.saturating_sub(50);
        }
        buf.make_mut_slice()[i] = Rgba8Pixel::new(r, g, b, p[3]);
    }
    buf
}

#[derive(Clone)]
pub struct ReliefPreviewContext {
    pub state: ReliefPreviewState,
    pub reply_tx: std::sync::mpsc::Sender<Option<f32>>,
}
