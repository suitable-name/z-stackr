//! Where alignment spends its time: one frame aligned onto another the way the
//! pipeline does it, plus the cost of one objective evaluation at every pyramid
//! level and of the final warp.
//!
//! ```text
//! cargo run --release -p z-stackr-pipeline --example align_profile -- sample/frame_0000.jpg sample/frame_0010.jpg
//! ```

use std::{path::PathBuf, time::Instant};

use stacker_align::{
    Matrix3,
    refine::{LumaPyramid, RegistrationParams, objective_at_level},
    transform::warp_image,
};
use stacker_core::settings::{AlignmentModeSetting, PreprocessingSettings};
use stacker_pipeline::{
    align::{AlignmentSettings, align_frame_with_pyramid},
    load::decode_frame,
};

fn main() {
    let args: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    assert!(args.len() == 2, "usage: align_profile <reference> <frame>");
    let pre = PreprocessingSettings::default();
    let reference = decode_frame(&args[0], &pre).expect("decode reference");
    let frame = decode_frame(&args[1], &pre).expect("decode frame");
    println!("frames {}x{}", reference.width, reference.height);

    let start = Instant::now();
    let pyramid = LumaPyramid::new(&reference);
    println!(
        "reference pyramid: {:.1} ms",
        start.elapsed().as_secs_f64() * 1e3
    );

    // One objective evaluation per level, at an identity-ish parameter vector.
    let src_pyramid = LumaPyramid::new(&frame);
    let mut params = RegistrationParams::identity().to_vec();
    params[0] += 0.001;
    params[1] -= 0.001;
    for level in 0..pyramid.n_levels() {
        let (r, w, h) = pyramid.level_luma(level, &reference.luma);
        let (s, _, _) = src_pyramid.level_luma(level, &frame.luma);
        let reps = if w * h > 1_000_000 { 5 } else { 50 };
        let start = Instant::now();
        let mut sink = 0.0;
        for _ in 0..reps {
            sink += objective_at_level(&params, r, s, w, h);
        }
        let per = start.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
        println!("level {level}: {w}x{h}, one evaluation {per:.3} ms (sink {sink:.3})");
    }

    for (name, mode, bounded) in [
        ("registration", AlignmentModeSetting::Registration, false),
        ("affine", AlignmentModeSetting::Affine, false),
        ("affine bounded", AlignmentModeSetting::Affine, true),
        ("translation", AlignmentModeSetting::Translation, false),
    ] {
        let settings = AlignmentSettings::new(mode, bounded);
        let start = Instant::now();
        let aligned =
            align_frame_with_pyramid(&reference, &pyramid, frame.clone(), &settings, None);
        println!(
            "align_frame ({name}): {:.0} ms, matrix {:?}",
            start.elapsed().as_secs_f64() * 1e3,
            aligned.matrix_applied.as_slice()
        );
    }

    let m = Matrix3::new(1.001, 0.0005, 2.5, -0.0005, 1.001, -1.5, 0.0, 0.0, 1.0);
    let start = Instant::now();
    let warped = warp_image(&frame, &m).expect("warp");
    println!(
        "warp_image: {:.0} ms ({} px)",
        start.elapsed().as_secs_f64() * 1e3,
        warped.luma.len()
    );
}
