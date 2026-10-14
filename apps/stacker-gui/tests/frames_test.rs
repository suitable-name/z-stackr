use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU32, Ordering},
};

use stacker_core::image::PlanarImage;
use stacker_gui::{
    frames::{FrameError, apply_order, assemble, decode_frames, working_paths},
    settings::PreprocessingSettings,
};

fn path(name: &str) -> PathBuf {
    PathBuf::from(name)
}

/// A frame whose first luma sample identifies it.
fn frame(w: usize, h: usize, marker: f32) -> PlanarImage<f32> {
    let mut image = PlanarImage::new(w, h);
    image.luma[0] = marker;
    image
}

fn marker_of(image: &PlanarImage<f32>) -> f32 {
    image.luma[0]
}

fn paths(names: &[&str]) -> Vec<PathBuf> {
    names.iter().map(|n| path(n)).collect()
}

#[test]
fn sorting_rewrites_the_list_in_the_order_of_the_census() {
    let list = paths(&["a", "b", "c", "d"]);
    let order = paths(&["c", "a", "d", "b"]);
    assert_eq!(apply_order(&list, &order), order);
}

#[test]
fn files_the_run_did_not_look_at_keep_their_places() {
    // `b` failed to decode and is not in the census order.
    let list = paths(&["a", "b", "c", "d"]);
    let order = paths(&["d", "c", "a"]);
    assert_eq!(apply_order(&list, &order), paths(&["d", "b", "c", "a"]));
}

#[test]
fn strangers_and_repeats_in_the_order_are_ignored() {
    let list = paths(&["a", "b"]);
    let order = paths(&["z", "b", "b", "a"]);
    assert_eq!(apply_order(&list, &order), paths(&["b", "a"]));
    assert_eq!(apply_order(&list, &[]), list);
    assert_eq!(apply_order(&[], &order), paths(&[]));
}

#[test]
fn a_failed_decode_drops_the_path_together_with_the_image() {
    let slots = vec![
        (path("a.png"), Ok(frame(4, 3, 1.0))),
        (path("bad.png"), Err("truncated file".to_owned())),
        (path("c.png"), Ok(frame(4, 3, 3.0))),
    ];
    let set = assemble(slots).unwrap();

    assert_eq!(set.paths, vec![path("a.png"), path("c.png")]);
    assert_eq!(set.len(), 2);
    // Position i of the paths is the file position i of the images came from.
    assert!((marker_of(&set.frames[0]) - 1.0).abs() < 1e-6);
    assert!((marker_of(&set.frames[1]) - 3.0).abs() < 1e-6);

    assert_eq!(set.skipped.len(), 1);
    assert_eq!(set.skipped[0].path, path("bad.png"));
    assert_eq!(set.skipped[0].reason, "truncated file");
    let note = set.skipped_note().unwrap();
    assert!(note.contains("bad.png"), "the note names the file: {note}");
}

#[test]
fn every_index_into_the_set_is_in_range_after_a_failure() {
    let slots = vec![
        (path("a.png"), Err("x".to_owned())),
        (path("b.png"), Ok(frame(2, 2, 2.0))),
        (path("c.png"), Ok(frame(2, 2, 3.0))),
    ];
    let set = assemble(slots).unwrap();
    // An auto-cull order over the decoded frames indexes the frames; the same
    // index must be valid in, and name the right file in, the path list.
    let order: Vec<usize> = (0..set.len()).rev().collect();
    let kept: Vec<&PathBuf> = order.iter().map(|&idx| &set.paths[idx]).collect();
    assert_eq!(set.paths.len(), set.frames.len());
    assert_eq!(kept, vec![&path("c.png"), &path("b.png")]);
    assert!((marker_of(&set.frames[order[0]]) - 3.0).abs() < 1e-6);
}

#[test]
fn a_frame_of_another_size_is_rejected_by_name() {
    let slots = vec![
        (path("a.png"), Ok(frame(4, 3, 1.0))),
        (path("rotated.png"), Ok(frame(3, 4, 2.0))),
    ];
    let error = assemble(slots).unwrap_err();
    assert_eq!(
        error,
        FrameError::DimensionMismatch {
            path: path("rotated.png"),
            reference: path("a.png"),
            expected: (4, 3),
            found: (3, 4),
        }
    );
    let message = error.to_string();
    assert!(message.contains("rotated.png"), "{message}");
    assert!(
        message.contains("3x4") && message.contains("4x3"),
        "{message}"
    );
}

#[test]
fn the_reference_is_the_first_frame_that_decoded() {
    let slots = vec![
        (path("a.png"), Err("unreadable".to_owned())),
        (path("b.png"), Ok(frame(4, 3, 1.0))),
        (path("c.png"), Ok(frame(5, 3, 2.0))),
    ];
    let error = assemble(slots).unwrap_err();
    let FrameError::DimensionMismatch {
        path: offender,
        reference,
        ..
    } = error
    else {
        panic!("expected a dimension mismatch");
    };
    assert_eq!(offender, path("c.png"));
    assert_eq!(reference, path("b.png"));
}

#[test]
fn nothing_decoding_is_an_error_that_says_why() {
    let error = assemble(vec![
        (path("a.png"), Err("not an image".to_owned())),
        (path("b.png"), Err("also not".to_owned())),
    ])
    .unwrap_err();
    assert_eq!(
        error,
        FrameError::NoFrames {
            attempted: 2,
            first_reason: Some("not an image".to_owned()),
        }
    );
    let message = error.to_string();
    assert!(message.contains("None of the 2"), "{message}");
    assert!(message.contains("not an image"), "{message}");
}

#[test]
fn an_empty_list_asks_for_images() {
    let error = assemble(Vec::new()).unwrap_err();
    assert!(error.to_string().contains("add some images"), "{error}");
}

#[test]
fn working_paths_drops_temporary_results_reverses_and_subsamples() {
    let paths = vec![
        path("one.png"),
        path("stacker_result_Apex_1.png"),
        path("two.png"),
        path("three.png"),
        path("four.png"),
    ];
    assert_eq!(
        working_paths(&paths, false, 1),
        vec![
            path("one.png"),
            path("two.png"),
            path("three.png"),
            path("four.png")
        ]
    );
    assert_eq!(
        working_paths(&paths, true, 1),
        vec![
            path("four.png"),
            path("three.png"),
            path("two.png"),
            path("one.png")
        ]
    );
    // Reverse first, then keep every second one.
    assert_eq!(
        working_paths(&paths, true, 2),
        vec![path("four.png"), path("two.png")]
    );
}

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "z-stackr-gui-frames-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn decoding_real_files_skips_the_broken_one_and_keeps_the_rest_aligned() {
    let dir = temp_dir("decode");
    let good_a = dir.join("a.png");
    let broken = dir.join("b.png");
    let good_c = dir.join("c.png");
    image::RgbImage::new(8, 6).save(&good_a).unwrap();
    fs::write(&broken, b"this is not a png").unwrap();
    image::RgbImage::new(8, 6).save(&good_c).unwrap();

    let set = decode_frames(
        vec![good_a.clone(), broken.clone(), good_c.clone()],
        &PreprocessingSettings::default(),
    )
    .unwrap();

    assert_eq!(set.paths, vec![good_a, good_c]);
    assert_eq!(set.frames.len(), 2);
    assert_eq!((set.frames[0].width, set.frames[0].height), (8, 6));
    assert_eq!(set.skipped.len(), 1);
    assert_eq!(set.skipped[0].path, broken);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn decoding_files_of_different_sizes_reports_the_odd_one() {
    let dir = temp_dir("mismatch");
    let wide = dir.join("wide.png");
    let narrow = dir.join("narrow.png");
    image::RgbImage::new(8, 6).save(&wide).unwrap();
    image::RgbImage::new(9, 6).save(&narrow).unwrap();

    let error = decode_frames(
        vec![wide, narrow.clone()],
        &PreprocessingSettings::default(),
    )
    .unwrap_err();
    let FrameError::DimensionMismatch { path, found, .. } = error else {
        panic!("expected a dimension mismatch");
    };
    assert_eq!(path, narrow);
    assert_eq!(found, (9, 6));
    let _ = fs::remove_dir_all(&dir);
}
