//! Where the GPU `Relief` spends its time: upload and fold per frame, the focus
//! map read-back, and the selection, smoothing and read-back at the end.
//!
//! ```text
//! cargo run --release -p z-stackr-gpu --example relief_phases -- 3840 2160 8
//! ```

use std::time::Instant;

use stacker_core::image::PlanarImage;
use stacker_gpu::{GpuContext, relief::ReliefPlan};

fn main() {
    let mut args = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse::<usize>().ok());
    let (w, h, n) = (
        args.next().unwrap_or(3840),
        args.next().unwrap_or(2160),
        args.next().unwrap_or(8),
    );
    let mut frame = PlanarImage::new(w, h);
    for (i, v) in frame.luma.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "a test pattern")]
        let x = (i % 977) as f32;
        *v = 0.1f32.mul_add((x * 0.37).sin(), 0.3);
    }
    let ctx = GpuContext::new(None).expect("a GPU");
    let start = Instant::now();
    let mut plan = ReliefPlan::new(&ctx, w, h).expect("plan");
    println!("plan: {:.0} ms", start.elapsed().as_secs_f64() * 1e3);
    for round in 0..2 {
        plan.begin(u32::try_from(n).expect("few"), 5)
            .expect("radius");
        let start = Instant::now();
        for _ in 0..n {
            plan.fold(&frame).expect("fold");
        }
        let fold = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        let max = plan.focus_max().expect("focus map");
        let read = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        let out = plan.finish(0.0, 2, 0.01).expect("finish");
        let finish = start.elapsed().as_secs_f64() * 1e3;
        println!(
            "round {round}: fold {n} frames {fold:.0} ms ({:.1} per frame), focus read-back {read:.0} ms, finish {finish:.0} ms [{} {}]",
            fold / f64::from(u32::try_from(n).expect("few")),
            max.len(),
            out.luma.len()
        );
    }
}
