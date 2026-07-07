use stacker_core::error::StackerError;
use std::io;

#[test]
fn test_error_display() {
    let err = StackerError::Io(io::Error::new(io::ErrorKind::NotFound, "file not found"));
    assert_eq!(format!("{err}"), "IO error: file not found");

    let err = StackerError::AlignmentFailed("too few points".to_string());
    assert_eq!(format!("{err}"), "Alignment failed: too few points");

    let err = StackerError::MathError("division by zero".to_string());
    assert_eq!(format!("{err}"), "Math error: division by zero");
}
