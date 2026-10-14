//! Frame discovery and selection: `order`.

use std::{cmp::Ordering, path::PathBuf};

use stacker_core::settings::StackingSettings;
use stacker_pipeline::{
    PipelineError,
    order::{
        FRAME_EXTENSIONS, collect_image_paths, has_frame_extension, is_excluded, is_result_file,
        natural_cmp, natural_sort, select_frames, select_frames_by,
    },
};

fn dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "stacker_pipeline_order_{tag}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn touch(dir: &std::path::Path, name: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, b"x").unwrap();
    p
}

fn names(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect()
}

fn list(n: usize) -> Vec<PathBuf> {
    (0..n).map(|i| PathBuf::from(format!("f{i}.png"))).collect()
}

#[test]
fn natural_order_puts_frame_2_before_frame_10() {
    assert_eq!(natural_cmp("frame_2.png", "frame_10.png"), Ordering::Less);
    assert_eq!(
        natural_cmp("frame_10.png", "frame_2.png"),
        Ordering::Greater
    );
    assert_eq!(natural_cmp("a9b", "a10b"), Ordering::Less);
    assert_eq!(natural_cmp("img_001", "img_1"), Ordering::Equal);
    assert_eq!(natural_cmp("B_1", "a_1"), Ordering::Greater);

    let mut paths: Vec<PathBuf> = ["x/f10.png", "x/f2.png", "x/F1.png", "x/f1.png"]
        .iter()
        .map(PathBuf::from)
        .collect();
    natural_sort(&mut paths);
    // The tie between `F1` and `f1` is broken by the plain path comparison.
    assert_eq!(
        names(&paths),
        ["F1.png", "f1.png", "f2.png", "f10.png"],
        "natural order, ties made total"
    );
}

#[test]
fn collection_sorts_naturally_and_skips_products_and_other_files() {
    let d = dir("collect");
    for name in [
        "frame_10.png",
        "frame_2.png",
        "frame_1.png",
        "notes.txt",
        "A.JPG",
    ] {
        touch(&d, name);
    }
    touch(&d, "stacker_result_Apex_17.png");
    let output = touch(&d, "stacked_result.png");
    std::fs::create_dir(d.join("sub.png")).unwrap();

    let paths = collect_image_paths(&d, std::slice::from_ref(&output)).unwrap();
    assert_eq!(
        names(&paths),
        ["A.JPG", "frame_1.png", "frame_2.png", "frame_10.png"],
        "extensions are matched case-insensitively; the output, the app's temporary \
         result, the text file and the directory are not frames"
    );

    // Without the exclusion the output is a frame like any other.
    let with_output = collect_image_paths(&d, &[]).unwrap();
    assert_eq!(with_output.len(), 5);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn an_empty_or_missing_directory_is_an_error() {
    let d = dir("empty");
    assert!(matches!(
        collect_image_paths(&d, &[]),
        Err(PipelineError::NoImages(_))
    ));
    let missing = d.join("nope");
    assert!(matches!(
        collect_image_paths(&missing, &[]),
        Err(PipelineError::Io(_))
    ));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn exclusion_matches_a_not_yet_existing_output() {
    let d = dir("exclude");
    let frame = touch(&d, "frame_1.png");
    let future = d.join("later.png");
    assert!(is_excluded(&future, std::slice::from_ref(&future)));
    assert!(!is_excluded(&frame, std::slice::from_ref(&future)));
    assert!(!is_excluded(&frame, &[]));
    // A relative spelling of the same file is the same file.
    let detour = d.join(".").join("frame_1.png");
    assert!(is_excluded(&detour, std::slice::from_ref(&frame)));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn extension_and_result_markers() {
    for ok in ["a.png", "a.JPG", "a.jpeg", "a.tif", "a.TIFF"] {
        assert!(has_frame_extension(std::path::Path::new(ok)), "{ok}");
    }
    for bad in ["a.txt", "a", "a.png.bak", "a.gif"] {
        assert!(!has_frame_extension(std::path::Path::new(bad)), "{bad}");
    }
    assert!(is_result_file(std::path::Path::new(
        "tmp/stacker_result_Relief_5.png"
    )));
    assert!(!is_result_file(std::path::Path::new("tmp/result.png")));
}

/// The exported list is the single source: every entry is a frame extension
/// (in either case), and the list is lower-case and dot-less so a picker
/// filter can use it as it is.
#[test]
fn frame_extensions_list_agrees_with_the_predicate() {
    assert!(FRAME_EXTENSIONS.contains(&"png") && FRAME_EXTENSIONS.contains(&"tiff"));
    for ext in FRAME_EXTENSIONS {
        assert_eq!(*ext, ext.to_ascii_lowercase());
        assert!(!ext.starts_with('.'));
        for spelled in [
            format!("a.{ext}"),
            format!("a.{}", ext.to_ascii_uppercase()),
        ] {
            assert!(
                has_frame_extension(std::path::Path::new(&spelled)),
                "{spelled}"
            );
        }
    }
}

/// `sort_reverse` reverses first, `stack_every_nth` samples the reversed list,
/// and result files never take part; the same rule for the CLI and the GUI.
#[test]
fn selection_reverses_then_takes_every_nth() {
    let mut settings = StackingSettings::default();
    assert_eq!(select_frames(list(7), &settings).len(), 7);

    settings.stack_every_nth = 3;
    assert_eq!(
        names(&select_frames(list(7), &settings)),
        ["f0.png", "f3.png", "f6.png"]
    );

    settings.preprocessing.sort_reverse = true;
    assert_eq!(
        names(&select_frames(list(7), &settings)),
        ["f6.png", "f3.png", "f0.png"]
    );

    settings.stack_every_nth = 0;
    settings.preprocessing.sort_reverse = false;
    assert_eq!(select_frames(list(4), &settings).len(), 4, "0 acts as 1");

    let mut with_result = list(3);
    with_result.insert(1, PathBuf::from("tmp/stacker_result_Apex_1.png"));
    assert_eq!(
        names(&select_frames_by(with_result, true, 2)),
        ["f2.png", "f0.png"],
        "the result file is dropped before reversing and sampling"
    );
}
