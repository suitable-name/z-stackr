//! Every `step`-th frame of a folder aligned onto its first frame with the
//! default settings, as the apps do, printing the transform each frame got. A
//! frame whose transform is the identity was left unaligned (the identity gate
//! or a fallback); on a focus series with breathing that shows as echoed edges.
//!
//! ```text
//! cargo run -p z-stackr-pipeline --features gpu --example align_stack -- sample 50 3 gpu
//! ```
//!
//! Arguments: the folder, the pre-resize percentage, the step, and `gpu` or `cpu`.

use std::{path::PathBuf, time::Instant};

use stacker_align::refine::{LumaPyramid, refined_beats_identity};
use stacker_core::settings::{ComputeDevice, StackingSettings};
use stacker_pipeline::{
    align::{AlignmentSettings, align_frame_with_compute},
    compute::Compute,
    load::decode_frame,
    order::collect_image_paths,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        args.len() == 4,
        "usage: align_stack <folder> <pre-resize %> <step> <gpu|cpu>"
    );
    let folder = PathBuf::from(&args[0]);
    let percent: u32 = args[1].parse().expect("pre-resize percentage");
    let step: usize = args[2].parse().expect("step");
    let mut settings = StackingSettings::default();
    settings.preprocessing.pre_resize_percent = percent;
    settings.compute_device = if args[3] == "gpu" {
        ComputeDevice::Gpu
    } else {
        ComputeDevice::Cpu
    };
    let paths = collect_image_paths(&folder, &[]).expect("list the folder");
    let align = AlignmentSettings::from(&settings);
    println!("{} frames, alignment {align:?}", paths.len());

    let reference = decode_frame(&paths[0], &settings.preprocessing).expect("decode reference");
    let pyramid = LumaPyramid::new(&reference);
    let mut compute = Compute::for_settings(&settings);
    for index in (step..paths.len()).step_by(step.max(1)) {
        let frame = decode_frame(&paths[index], &settings.preprocessing).expect("decode frame");
        let start = Instant::now();
        let aligned =
            align_frame_with_compute(&reference, &pyramid, frame.clone(), &align, None, &mut compute);
        let m = aligned.matrix_applied;
        // The gate's verdict on the transform the frame got (the refiners' matrix
        // is the inverse of the applied one).
        let gate = m
            .try_inverse()
            .is_some_and(|inverse| refined_beats_identity(&reference, &frame, &inverse));
        println!(
            "frame {index:>2}: scale {:.4} {:.4}  shift {:>8.2} {:>8.2}  fallback {:?}  covered {:.3}  gate keeps it {gate}  {:.0} ms",
            m[(0, 0)],
            m[(1, 1)],
            m[(0, 2)],
            m[(1, 2)],
            aligned.fallback,
            aligned.covered_fraction(),
            start.elapsed().as_secs_f64() * 1e3,
        );
    }
}
