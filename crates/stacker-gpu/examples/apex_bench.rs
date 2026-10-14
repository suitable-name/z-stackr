//! Times `Apex` on the CPU and on the GPU for the same synthetic stack.
//!
//! ```text
//! cargo run --release -p z-stackr-gpu --example apex_bench -- \
//!     --width 1536 --height 1536 --frames 50 --levels 8 --reps 3 --check
//! ```
//!
//! Options: `--width`, `--height`, `--frames`, `--levels`, `--reps` (timed
//! repetitions after one warm-up), `--adapter` (index from the printed list),
//! `--color`, `--no-grit`, `--check` (compare the two results), `--cpu-only`,
//! `--gpu-only`.
//!
//! The GPU time is split into what a real run pays: the end-to-end time (upload,
//! compute, download), then the parts on their own, so the transfer share is
//! visible.

#![expect(
    clippy::suboptimal_flops,
    reason = "the synthetic-image formulas read better without mul_add"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "pixel indices and counts are far below 2^24 (images) or 2^52 (statistics)"
)]

use std::time::{Duration, Instant};

use rayon::prelude::*;
use stacker_algo::apex::fuse::build_and_fuse_pyramids;
use stacker_core::image::PlanarImage;
use stacker_gpu::{GpuContext, apex::ApexPlan};

#[expect(
    clippy::struct_excessive_bools,
    reason = "a command-line option set: one flag per switch"
)]
struct Args {
    width: usize,
    height: usize,
    frames: usize,
    levels: usize,
    reps: usize,
    adapter: Option<usize>,
    use_color: bool,
    grit: bool,
    check: bool,
    cpu: bool,
    gpu: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        width: 1536,
        height: 1536,
        frames: 20,
        levels: 8,
        reps: 3,
        adapter: None,
        use_color: false,
        grit: true,
        check: false,
        cpu: true,
        gpu: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut number = |name: &str| -> usize {
            it.next()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("{name} needs a number"))
        };
        match flag.as_str() {
            "--width" => a.width = number("--width"),
            "--height" => a.height = number("--height"),
            "--frames" => a.frames = number("--frames"),
            "--levels" => a.levels = number("--levels"),
            "--reps" => a.reps = number("--reps"),
            "--adapter" => a.adapter = Some(number("--adapter")),
            "--color" => a.use_color = true,
            "--no-grit" => a.grit = false,
            "--check" => a.check = true,
            "--cpu-only" => a.gpu = false,
            "--gpu-only" => a.cpu = false,
            other => panic!("unknown option {other}"),
        }
    }
    a
}

fn hash(x: usize, y: usize, seed: u32) -> f32 {
    let mut h = u32::try_from(x & 0xFFFF_FFFF)
        .unwrap_or(0)
        .wrapping_mul(0x9E37_79B1)
        ^ u32::try_from(y & 0xFFFF_FFFF)
            .unwrap_or(0)
            .wrapping_mul(0x85EB_CA77)
        ^ seed.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    (h >> 8) as f32 / 16_777_216.0
}

/// Frame `k` of `n`: smooth content, with fine detail strongest where the depth
/// is near the frame's focus position.
fn frame(k: usize, n: usize, w: usize, h: usize) -> PlanarImage<f32> {
    let focus = (k as f32 + 0.5) / n as f32;
    let spread = 2.0 * (n * n) as f32;
    let mut luma = vec![0.0_f32; w * h];
    let mut ca = vec![0.0_f32; w * h];
    let mut cb = vec![0.0_f32; w * h];
    luma.par_chunks_mut(w)
        .zip(ca.par_chunks_mut(w))
        .zip(cb.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, ((rl, ra), rb))| {
            for x in 0..w {
                let depth = 0.5 + 0.5 * (0.013 * x as f32).sin() * (0.011 * y as f32).cos();
                let sharp = (-(depth - focus) * (depth - focus) * spread).exp();
                rl[x] = 0.25 + 0.4 * depth + (hash(x, y, 1) - 0.5) * (0.02 + 0.6 * sharp);
                ra[x] = 0.03 * (0.05 * x as f32).sin() + (hash(x, y, 2) - 0.5) * 0.2 * sharp;
                rb[x] = 0.03 * (0.07 * y as f32).cos() + (hash(x, y, 3) - 0.5) * 0.2 * sharp;
            }
        });
    PlanarImage::from_planes(w, h, luma, ca, cb).expect("planes have w * h samples")
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Run `f` once as a warm-up and `reps` times timed; the median in milliseconds.
fn timed<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut last = f();
    let mut times = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        last = f();
        times.push(ms(t.elapsed()));
    }
    (median(times), last)
}

#[expect(
    clippy::too_many_lines,
    reason = "a straight benchmark script: setup, CPU, GPU phases, report"
)]
fn main() {
    let a = parse_args();
    println!("adapters:");
    for s in GpuContext::adapters() {
        println!(
            "  [{}] {} ({}, {})",
            s.index, s.name, s.backend, s.device_type
        );
    }
    println!(
        "stack: {} frames of {}x{} ({:.1} MP), {} levels, color={}, grit={}",
        a.frames,
        a.width,
        a.height,
        (a.width * a.height) as f64 / 1e6,
        a.levels,
        a.use_color,
        a.grit
    );
    println!(
        "CPU build: avx2={}, avx512f={}, rayon threads={}",
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx512f"),
        rayon::current_num_threads()
    );

    let t = Instant::now();
    let frames: Vec<_> = (0..a.frames)
        .map(|k| frame(k, a.frames, a.width, a.height))
        .collect();
    println!("frames generated in {:.0} ms", ms(t.elapsed()));

    let mut cpu_result = None;
    let mut cpu_ms = f64::NAN;
    if a.cpu {
        let (m, img) = timed(a.reps, || {
            build_and_fuse_pyramids(&frames, a.levels, a.use_color, a.grit).reconstruct()
        });
        cpu_ms = m;
        cpu_result = Some(img);
        println!(
            "CPU   {:>9.1} ms  ({:.1} ms per frame)",
            m,
            m / a.frames as f64
        );
    }

    if !a.gpu {
        return;
    }
    let ctx = match GpuContext::new(a.adapter) {
        Ok(c) => c,
        Err(e) => {
            println!("GPU unavailable: {e}");
            return;
        }
    };
    let info = ctx.info();
    println!(
        "GPU   {} ({:?}, {:?}); max buffer {} MiB",
        info.name,
        info.backend,
        info.device_type,
        ctx.limits().max_buffer_size >> 20
    );
    let t = Instant::now();
    let mut plan = match ApexPlan::new(&ctx, a.width, a.height, a.levels, a.use_color, a.grit) {
        Ok(p) => p,
        Err(e) => {
            println!("GPU plan failed: {e}");
            return;
        }
    };
    println!(
        "plan built in {:.0} ms, {} levels, {} MiB of device memory (+ one image for read-back)",
        ms(t.elapsed()),
        plan.levels(),
        plan.device_bytes() >> 20
    );
    let n = u32::try_from(a.frames).expect("frames fit u32");

    // End to end: what a real run pays.
    let (e2e, gpu_result) = timed(a.reps, || {
        plan.begin(n);
        for f in &frames {
            plan.fold(f).expect("fold");
        }
        plan.finish().expect("finish")
    });

    // Compute only: the first frame stays resident and is folded n times.
    plan.upload(&frames[0]).expect("upload");
    plan.flush();
    let (compute, ()) = timed(a.reps, || {
        plan.begin(n);
        for _ in 0..a.frames {
            plan.fold_uploaded();
        }
        plan.reconstruct();
    });

    // Upload only: every frame crosses to the device, nothing is computed.
    let (upload, ()) = timed(a.reps, || {
        for f in &frames {
            plan.upload(f).expect("upload");
            plan.flush();
        }
    });

    // Download only.
    let (download, _) = timed(a.reps, || plan.read_back().expect("read back"));

    println!(
        "GPU end to end {e2e:>9.1} ms  ({:.1} ms per frame)",
        e2e / a.frames as f64
    );
    println!(
        "  compute only {compute:>9.1} ms  ({:.1} ms per frame, includes reconstruction)",
        compute / a.frames as f64
    );
    println!(
        "  upload only  {upload:>9.1} ms  ({:.1} ms per frame)",
        upload / a.frames as f64
    );
    println!("  download     {download:>9.1} ms");
    if cpu_ms.is_finite() {
        println!(
            "speed-up end to end: {:.2}x   compute only: {:.2}x",
            cpu_ms / e2e,
            cpu_ms / compute
        );
    }

    if a.check
        && let Some(cpu) = cpu_result
    {
        let mut max = 0.0_f32;
        let mut sum = 0.0_f64;
        let mut visible = 0_usize;
        let mut samples = 0_usize;
        for (pc, pg) in [
            (&cpu.luma, &gpu_result.luma),
            (&cpu.chroma_a, &gpu_result.chroma_a),
            (&cpu.chroma_b, &gpu_result.chroma_b),
        ] {
            for (x, y) in pc.iter().zip(pg) {
                let d = (x - y).abs();
                max = max.max(d);
                sum += f64::from(d);
                visible += usize::from(d > 1e-3);
                samples += 1;
            }
        }
        println!(
            "check: max difference {max:.2e}, mean {:.2e}, {visible} of {samples} samples differ by more than 1e-3",
            sum / samples as f64
        );
    }
}
