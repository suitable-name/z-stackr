//! Build every pipeline of the alignment and `Relief` shaders on one adapter and
//! report each step, to find which one a driver refuses.
//!
//! ```text
//! cargo run --release -p z-stackr-gpu --example shader_probe -- [adapter index]
//! ```

use std::sync::Arc;

use stacker_gpu::{
    GpuContext,
    align::{ObjectivePlan, Sampler, WarpPlan},
    relief::ReliefPlan,
};

fn main() {
    for a in GpuContext::adapters() {
        eprintln!(
            "adapter {}: {} ({}, {})",
            a.index, a.name, a.backend, a.device_type
        );
    }
    let index = std::env::args().nth(1).and_then(|s| s.parse().ok());
    let ctx: Arc<GpuContext> = GpuContext::new(index).expect("adapter");
    eprintln!("using {:?}", ctx.info().name);
    let (w, h) = (64, 48);
    let plane = vec![0.5_f32; w * h];
    eprintln!("objective plan...");
    let mut plan =
        ObjectivePlan::new(&ctx, &plane, &plane, w, h, Sampler::Bicubic).expect("objective plan");
    eprintln!("moments...");
    let m = plan
        .moments(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        .expect("moments");
    eprintln!("moments {m:?}");
    eprintln!("warp plan...");
    let _warp = WarpPlan::new(&ctx, w, h).expect("warp plan");
    eprintln!("relief plan...");
    let _relief = ReliefPlan::new(&ctx, w, h).expect("relief plan");
    eprintln!("done");
}
