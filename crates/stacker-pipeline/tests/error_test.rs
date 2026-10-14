//! `PipelineError` and the small shared helpers.

use std::path::PathBuf;

use stacker_core::error::StackerError;
use stacker_pipeline::{
    Cancel, NeverCancel, NoProgress, PipelineError, Progress, Stage, fraction_of,
};

#[test]
fn messages_name_the_files_and_sizes() {
    let mismatch = PipelineError::SizeMismatch {
        path: PathBuf::from("in/frame_01.png"),
        reference: PathBuf::from("in/frame_00.png"),
        expected: (64, 48),
        found: (54, 48),
    };
    let text = mismatch.to_string();
    assert!(
        text.contains("frame_01.png") && text.contains("54x48"),
        "{text}"
    );
    assert!(
        text.contains("frame_00.png") && text.contains("64x48"),
        "{text}"
    );

    let decode = PipelineError::decode(std::path::Path::new("in/bad.png"), "truncated");
    assert!(decode.to_string().contains("bad.png"));
    assert!(decode.to_string().contains("truncated"));
    assert!(
        PipelineError::NoImages(PathBuf::from("empty"))
            .to_string()
            .contains("empty")
    );
}

/// The CLI keeps reporting the same `StackerError` variants as before the
/// stages were shared.
#[test]
fn conversion_to_stacker_error_keeps_the_variants() {
    let no_images: StackerError = PipelineError::NoImages(PathBuf::from("d")).into();
    assert!(matches!(no_images, StackerError::AlignmentFailed(_)));

    let error = PipelineError::decode(std::path::Path::new("a.png"), "x");
    let text = error.to_string();
    let converted: StackerError = error.into();
    match converted {
        StackerError::Io(io) => assert_eq!(io.to_string(), text),
        other => panic!("expected an I/O error, got {other:?}"),
    }

    // A rejected request is a validation error, not an I/O error.
    let invalid: StackerError = PipelineError::invalid("bad option").into();
    assert!(matches!(&invalid, StackerError::Validation(m) if m == "bad option"));
    assert_eq!(invalid.to_string(), "Invalid request: bad option");

    let inner = StackerError::MathError("boom".to_owned());
    let wrapped: StackerError = PipelineError::Core(inner).into();
    assert!(matches!(wrapped, StackerError::MathError(m) if m == "boom"));

    let io: StackerError = PipelineError::from(std::io::Error::other("disk")).into();
    assert!(matches!(io, StackerError::Io(_)));
}

#[test]
fn progress_and_cancel_defaults_are_inert() {
    NoProgress.stage(Stage::Align, 0.5, "label");
    assert!(!NeverCancel.is_cancelled());
}

#[test]
fn fractions_are_clamped_and_total_zero_is_done() {
    assert!((fraction_of(1, 4) - 0.25).abs() < 1e-6);
    assert!((fraction_of(9, 4) - 1.0).abs() < 1e-6);
    assert!((fraction_of(0, 0) - 1.0).abs() < 1e-6);
}
