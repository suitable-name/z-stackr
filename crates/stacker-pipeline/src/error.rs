//! The error type of the shared stages.

use std::path::{Path, PathBuf};

use stacker_core::error::StackerError;

/// Why a shared stage failed.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// A frame could not be opened or decoded.
    #[error("cannot decode frame '{}': {message}", .path.display())]
    Decode {
        /// The frame that failed.
        path: PathBuf,
        /// The decoder's message.
        message: String,
    },

    /// A directory holds no frame.
    #[error("no images found in '{}'", .0.display())]
    NoImages(PathBuf),

    /// A frame is not the size of the reference frame.
    #[error(
        "frame '{}' is {}x{} but the reference frame '{}' is {}x{}; all frames of a stack must \
         have the same size",
        .path.display(), .found.0, .found.1, .reference.display(), .expected.0, .expected.1
    )]
    SizeMismatch {
        /// The offending frame.
        path: PathBuf,
        /// The reference frame the size was taken from.
        reference: PathBuf,
        /// The reference size as `(width, height)`.
        expected: (usize, usize),
        /// The offending frame's size as `(width, height)`.
        found: (usize, usize),
    },

    /// A request that cannot be honoured (bad option, bad input set).
    #[error("{0}")]
    Invalid(String),

    /// The caller's [`Cancel`](crate::Cancel) asked to stop.
    #[error("cancelled")]
    Cancelled,

    /// A core-library failure (tile store, matrix maths).
    #[error(transparent)]
    Core(#[from] StackerError),

    /// A file-system failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// An image encoder or decoder failure.
    #[error(transparent)]
    Image(#[from] image::ImageError),
}

impl PipelineError {
    /// A decode failure for `path`.
    #[must_use]
    pub fn decode(path: &Path, message: impl std::fmt::Display) -> Self {
        Self::Decode {
            path: path.to_path_buf(),
            message: message.to_string(),
        }
    }

    /// A request that cannot be honoured.
    #[must_use]
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

impl From<PipelineError> for StackerError {
    /// A rejected request becomes [`StackerError::Validation`]; the other
    /// variants map to the [`StackerError`] variant of the same failure (I/O,
    /// image, alignment) so the apps' error handling is unchanged.
    fn from(error: PipelineError) -> Self {
        match error {
            PipelineError::NoImages(_) => Self::AlignmentFailed(error.to_string()),
            PipelineError::Decode { .. } | PipelineError::SizeMismatch { .. } => Self::Io(
                std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()),
            ),
            PipelineError::Invalid(message) => Self::Validation(message),
            PipelineError::Cancelled => Self::Io(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                error.to_string(),
            )),
            PipelineError::Core(e) => e,
            PipelineError::Io(e) => Self::Io(e),
            PipelineError::Image(e) => Self::Image(e),
        }
    }
}
