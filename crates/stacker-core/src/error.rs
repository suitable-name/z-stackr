//! The error type shared by every crate of the workspace.

use std::fmt;

/// Everything that can go wrong in the stacking pipeline, in one enum so the
/// crates can pass failures up without converting between error types.
///
/// `Io` and `Image` wrap the underlying error and expose it through
/// [`std::error::Error::source`]; the other variants carry a human-readable
/// message that is already complete (it is printed after a fixed prefix).
#[derive(Debug)]
pub enum StackerError {
    /// A file or directory operation failed (read, write, create, remove).
    Io(std::io::Error),
    /// The `image` crate could not decode or encode a picture.
    Image(image::ImageError),
    /// No tile was stored for the requested frame and rectangle; carries the
    /// rectangle's `(start_x, start_y)`.
    TileNotFound(usize, usize),
    /// Aligning a frame to the reference failed (too few matches, a singular
    /// fit, a non-finite matrix).
    AlignmentFailed(String),
    /// A numeric precondition was violated (mismatched plane lengths, an
    /// overflowing pixel count, a division that cannot be formed).
    MathError(String),
    /// A setting, option or settings file that cannot be used (unknown
    /// fusion mode, unreadable or invalid config file, missing AI model).
    Config(String),
    /// A request or input set that cannot be honoured (no frames, an unusable
    /// output path, a bad tile size).
    Validation(String),
}

impl std::error::Error for StackerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Image(e) => Some(e),
            _ => None,
        }
    }
}

impl fmt::Display for StackerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "IO error: {err}"),
            Self::Image(err) => write!(f, "Image error: {err}"),
            Self::TileNotFound(x, y) => write!(f, "Tile not found at ({x}, {y})"),
            Self::AlignmentFailed(msg) => write!(f, "Alignment failed: {msg}"),
            Self::MathError(msg) => write!(f, "Math error: {msg}"),
            Self::Config(msg) => write!(f, "Configuration error: {msg}"),
            Self::Validation(msg) => write!(f, "Invalid request: {msg}"),
        }
    }
}

impl From<std::io::Error> for StackerError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<image::ImageError> for StackerError {
    fn from(err: image::ImageError) -> Self {
        Self::Image(err)
    }
}
