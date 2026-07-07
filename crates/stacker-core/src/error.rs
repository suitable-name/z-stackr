#![allow(clippy::use_self, clippy::uninlined_format_args)]

use std::fmt;

#[derive(Debug)]
pub enum StackerError {
    Io(std::io::Error),
    Image(image::ImageError),
    AlignmentFailed(String),
    MathError(String),
}

impl std::error::Error for StackerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StackerError::Io(e) => Some(e),
            StackerError::Image(e) => Some(e),
            _ => None,
        }
    }
}

impl fmt::Display for StackerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StackerError::Io(err) => write!(f, "IO error: {}", err),
            StackerError::Image(err) => write!(f, "Image error: {}", err),
            StackerError::AlignmentFailed(msg) => write!(f, "Alignment failed: {}", msg),
            StackerError::MathError(msg) => write!(f, "Math error: {}", msg),
        }
    }
}

impl From<std::io::Error> for StackerError {
    fn from(err: std::io::Error) -> Self {
        StackerError::Io(err)
    }
}

impl From<image::ImageError> for StackerError {
    fn from(err: image::ImageError) -> Self {
        StackerError::Image(err)
    }
}

impl From<crate::io::LoadError> for StackerError {
    fn from(err: crate::io::LoadError) -> Self {
        match err {
            crate::io::LoadError::Image { source, .. } => StackerError::Image(source),
            other @ crate::io::LoadError::RawSupportDisabled { .. } => {
                StackerError::MathError(other.to_string())
            }
            #[cfg(feature = "raw")]
            other @ crate::io::LoadError::RawDecode { .. } => {
                StackerError::MathError(other.to_string())
            }
        }
    }
}
