//! `io`: RAW extension recognition and the decoder's two builds.

use std::path::Path;

use stacker_core::io::{
    RAW_EXTENSIONS, RAW_SUPPORTED, RawError, decode_raw_file, has_raw_extension, is_raw_extension,
};

#[test]
fn raw_extensions_are_recognised_in_any_letter_case() {
    for ext in [
        "cr2", "NEF", "Arw", "dng", "raf", "orf", "rw2", "pef", "cr3", "CR3",
    ] {
        assert!(is_raw_extension(ext), "{ext}");
    }
    for ext in [
        "png", "jpg", "jpeg", "tif", "tiff", "PNG", "", "txt", ".nef",
    ] {
        assert!(!is_raw_extension(ext), "{ext}");
    }
}

#[test]
fn the_extension_list_is_lower_case_dot_less_and_unique() {
    for (i, ext) in RAW_EXTENSIONS.iter().enumerate() {
        assert_eq!(*ext, ext.to_ascii_lowercase());
        assert!(!ext.starts_with('.'));
        assert!(!RAW_EXTENSIONS[..i].contains(ext), "{ext} listed twice");
    }
}

#[test]
fn paths_are_judged_by_their_extension() {
    assert!(has_raw_extension(Path::new("shots/IMG_0001.CR2")));
    assert!(has_raw_extension(Path::new("a.b.dng")));
    assert!(!has_raw_extension(Path::new("frame.tif")));
    assert!(!has_raw_extension(Path::new("nef")));
    assert!(!has_raw_extension(Path::new("")));
}

#[test]
fn without_raw_support_a_raw_path_is_a_clear_error() {
    let result = decode_raw_file(Path::new("this_file_does_not_exist.NEF"));
    if RAW_SUPPORTED {
        // The feature is on: the missing file is a decode failure, not "disabled".
        assert!(matches!(result, Err(RawError::Decode { .. })), "{result:?}");
    } else {
        let err = result.unwrap_err();
        assert_eq!(
            err,
            RawError::Disabled {
                ext: "NEF".to_owned()
            }
        );
        let message = err.to_string();
        assert!(message.contains("--features raw"), "{message}");
        assert!(message.contains("NEF"), "{message}");
    }
}

#[test]
fn raw_support_follows_the_build_configuration() {
    assert_eq!(
        RAW_SUPPORTED,
        cfg!(all(feature = "raw", not(target_arch = "wasm32")))
    );
}
