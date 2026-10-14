//! CPU against GPU for the stages with GPU kernels, on real frames: one objective
//! evaluation per pyramid level, the alignment of one frame per mode, the final
//! warp, and the `Relief` fusion of the first `count` frames.
//!
//! ```text
//! cargo run --release -p z-stackr-pipeline --features gpu --example gpu_profile -- sample 8
//! ```

use std::{path::PathBuf, sync::Arc, time::Instant};

use stacker_align::{
    Matrix3,
    refine::{
        LumaPyramid, RegistrationParams, matrix_to_params, objective_at_level, params_to_matrix,
    },
    transform::warp_image,
};
use stacker_core::{
    image::PlanarImage,
    settings::{AlignmentModeSetting, ComputeDevice, PreprocessingSettings, StackingSettings},
};
use stacker_gpu::{
    GpuContext,
    align::{ObjectivePlan, Sampler},
};
use stacker_pipeline::{
    align::{AlignmentSettings, align_frame_with_compute, align_frame_with_pyramid},
    compute::Compute,
    fuse::{self, relief_threshold},
    load::decode_frame,
};

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().unwrap_or_else(|| "sample".to_owned()));
    let count: usize = args.next().and_then(|arg| arg.parse().ok()).unwrap_or(8);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("a folder of frames")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("jpg"))
        })
        .collect();
    paths.sort();
    paths.truncate(count.max(2));
    let pre = PreprocessingSettings::default();
    let frames: Vec<PlanarImage<f32>> = paths
        .iter()
        .map(|path| decode_frame(path, &pre).expect("decode"))
        .collect();
    println!(
        "{} frames of {}x{}",
        frames.len(),
        frames[0].width,
        frames[0].height
    );

    let ctx: Arc<GpuContext> = GpuContext::new(None).expect("a GPU");
    println!("GPU: {}", ctx.info().name);

    let reference = &frames[0];
    let source = &frames[frames.len() / 2];
    let ref_pyramid = LumaPyramid::new(reference);
    objective_per_level(&ctx, reference, source, &ref_pyramid);

    let gpu_settings = StackingSettings {
        compute_device: ComputeDevice::Gpu,
        ..StackingSettings::default()
    };
    align_one_frame(reference, source, &ref_pyramid, &gpu_settings);
    final_warp(source, &gpu_settings);
    relief_fusion(&frames, &gpu_settings);
}

// ── One objective evaluation per level ───────────────────────────────────────

fn objective_per_level(
    ctx: &Arc<GpuContext>,
    reference: &PlanarImage<f32>,
    source: &PlanarImage<f32>,
    ref_pyramid: &LumaPyramid,
) {
    let src_pyramid = LumaPyramid::new(source);
    let mut params = RegistrationParams::identity().to_vec();
    params[0] += 0.001;
    params[1] -= 0.001;
    for level in 0..ref_pyramid.n_levels() {
        let (ref_luma, lw, lh) = ref_pyramid.level_luma(level, &reference.luma);
        let (src_luma, _, _) = src_pyramid.level_luma(level, &source.luma);
        let reps = if lw * lh > 1_000_000 { 10 } else { 100 };
        let start = Instant::now();
        let mut sink = 0.0;
        for _ in 0..reps {
            sink += objective_at_level(&params, ref_luma, src_luma, lw, lh);
        }
        let cpu = ms(start) / f64::from(reps);
        let start = Instant::now();
        let mut plan =
            ObjectivePlan::new(ctx, ref_luma, src_luma, lw, lh, Sampler::Bicubic).expect("plan");
        let upload = ms(start);
        let registration = RegistrationParams {
            tx: params[0],
            ty: params[1],
            scale: params[2],
            rotate: params[3],
        };
        let inverse = params_to_matrix(&registration, lw, lh)
            .and_then(Matrix3::try_inverse)
            .expect("invertible");
        let rows: [[f32; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| inverse[(i, j)]));
        let _ = plan.moments(&rows).expect("warm-up");
        let start = Instant::now();
        for _ in 0..reps {
            let moments = plan.moments(&rows).expect("moments");
            #[expect(clippy::cast_precision_loss, reason = "a pixel count")]
            let n = moments.count as f64;
            sink += moments.m2 / n;
        }
        let gpu = ms(start) / f64::from(reps);
        println!(
            "level {level}: {lw}x{lh}: CPU {cpu:.3} ms, GPU {gpu:.3} ms (upload {upload:.1} ms) [{sink:.1}]"
        );
    }
}

// ── Alignment of one frame ───────────────────────────────────────────────────

fn align_one_frame(
    reference: &PlanarImage<f32>,
    source: &PlanarImage<f32>,
    ref_pyramid: &LumaPyramid,
    gpu_settings: &StackingSettings,
) {
    let (width, height) = (reference.width, reference.height);
    for (name, mode, bounded) in [
        ("affine", AlignmentModeSetting::Affine, false),
        ("affine bounded", AlignmentModeSetting::Affine, true),
        ("translation", AlignmentModeSetting::Translation, false),
        ("registration", AlignmentModeSetting::Registration, false),
    ] {
        let settings = AlignmentSettings::new(mode, bounded);
        let start = Instant::now();
        let pyramid_out =
            align_frame_with_pyramid(reference, ref_pyramid, source.clone(), &settings, None);
        let cpu_ms = ms(start);
        let mut compute = Compute::for_settings(gpu_settings);
        let start = Instant::now();
        let compute_out = align_frame_with_compute(
            reference,
            ref_pyramid,
            source.clone(),
            &settings,
            None,
            &mut compute,
        );
        let gpu_ms = ms(start);
        let (cpu_matrix, gpu_matrix) = (pyramid_out.matrix_applied, compute_out.matrix_applied);
        // How good each optimum is by the CPU objective at full resolution.
        let score = |applied: &Matrix3<f32>| {
            let inverse = applied.try_inverse().expect("invertible");
            let params = matrix_to_params(&inverse, width, height).to_vec();
            objective_at_level(&params, &reference.luma, &source.luma, width, height)
        };
        println!(
            "  objective at the optimum: CPU {:.7}, GPU {:.7}",
            score(&cpu_matrix),
            score(&gpu_matrix)
        );
        println!(
            "align {name}: CPU {cpu_ms:.0} ms, GPU {gpu_ms:.0} ms; shift CPU ({:.4}, {:.4}) GPU ({:.4}, {:.4}){}",
            cpu_matrix[(0, 2)],
            cpu_matrix[(1, 2)],
            gpu_matrix[(0, 2)],
            gpu_matrix[(1, 2)],
            compute
                .take_notice()
                .map_or_else(String::new, |notice| format!(" NOTICE {notice}"))
        );
    }
}

// ── Final warp ───────────────────────────────────────────────────────────────

fn final_warp(source: &PlanarImage<f32>, gpu_settings: &StackingSettings) {
    let matrix = Matrix3::new(1.001, 0.0005, 2.5, -0.0005, 1.001, -1.5, 0.0, 0.0, 1.0);
    let start = Instant::now();
    let _ = warp_image(source, &matrix).expect("warp");
    let cpu = ms(start);
    let mut compute = Compute::for_settings(gpu_settings);
    let _ = compute.warp(source, &matrix, false).expect("warm-up");
    let start = Instant::now();
    let _ = compute.warp(source, &matrix, false).expect("warp");
    println!("warp: CPU {cpu:.0} ms, GPU {:.0} ms", ms(start));
}

// ── Relief ───────────────────────────────────────────────────────────────────

fn relief_fusion(frames: &[PlanarImage<f32>], gpu_settings: &StackingSettings) {
    let settings = StackingSettings::default();
    let start = Instant::now();
    let cpu = fuse::relief(frames, &settings);
    let cpu_ms = ms(start);
    let mut compute = Compute::for_settings(gpu_settings);
    let start = Instant::now();
    let gpu = compute
        .relief(frames, gpu_settings, |map| {
            Some(relief_threshold(&map.luma, &settings).value)
        })
        .expect("not cancelled");
    let gpu_ms = ms(start);
    let diffs: Vec<f32> = cpu
        .luma
        .iter()
        .zip(&gpu.luma)
        .map(|(on_cpu, on_gpu)| (on_cpu - on_gpu).abs())
        .collect();
    let worst = diffs.iter().copied().fold(0.0_f32, f32::max);
    #[expect(clippy::cast_precision_loss, reason = "pixel counts")]
    let mean = diffs.iter().map(|&d| f64::from(d)).sum::<f64>() / diffs.len() as f64;
    let visible = diffs.iter().filter(|&&d| d > 1e-3).count();
    println!(
        "Relief of {} frames: CPU {cpu_ms:.0} ms, GPU {gpu_ms:.0} ms; luma difference largest {worst:.2e}, mean {mean:.2e}, above 1e-3 at {visible} of {} pixels",
        frames.len(),
        diffs.len()
    );
}
