//! Building the frame list for the Align and Stack passes.
//!
//! The list is built so that paths and decoded images can never drift apart: a
//! frame that fails to decode is dropped *together with* its path
//! ([`FrameSet::paths`] and [`FrameSet::frames`] are parallel), and a frame whose
//! size differs from the reference ends the run with an error that names the
//! file. Every index used afterwards (auto-cull order, aligned cache, status
//! counts) is a position in this list.

use std::{
    fmt,
    path::{Path, PathBuf},
};

use rayon::prelude::*;
use stacker_core::image::PlanarImage;
use stacker_pipeline::{PipelineError, load::decode_frame, order::select_frames_by};

use crate::settings::{PreprocessingSettings, StackingSettings};

/// A frame that was left out of the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedFrame {
    /// The file that was left out.
    pub path: PathBuf,
    /// The decoder's message.
    pub reason: String,
}

/// The decoded frames of one run. `paths[i]` is the file `frames[i]` was
/// decoded from; `frames[0]` is the alignment reference.
#[derive(Debug, Default)]
pub struct FrameSet {
    /// The files, parallel to `frames`.
    pub paths: Vec<PathBuf>,
    /// The decoded frames.
    pub frames: Vec<PlanarImage<f32>>,
    /// The files that could not be decoded and were left out.
    pub skipped: Vec<SkippedFrame>,
}

/// Why no usable frame list could be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    /// Nothing decoded: `attempted` files were tried, `first_reason` is the
    /// first decoder message (if any file was tried at all).
    NoFrames {
        /// How many files were tried.
        attempted: usize,
        /// The first decoder message.
        first_reason: Option<String>,
    },
    /// A frame is not the same size as the reference frame.
    DimensionMismatch {
        /// The frame with the wrong size.
        path: PathBuf,
        /// The reference frame.
        reference: PathBuf,
        /// The reference size `(width, height)`.
        expected: (usize, usize),
        /// The size of `path`.
        found: (usize, usize),
    },
}

fn file_label(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoFrames {
                attempted: 0,
                first_reason: _,
            } => write!(f, "No images to process: add some images first."),
            Self::NoFrames {
                attempted,
                first_reason,
            } => {
                write!(f, "None of the {attempted} image(s) could be loaded")?;
                if let Some(reason) = first_reason {
                    write!(f, " ({reason})")?;
                }
                write!(f, ".")
            }
            Self::DimensionMismatch {
                path,
                reference,
                expected,
                found,
            } => write!(
                f,
                "{} is {}x{} px but the reference frame {} is {}x{} px; all frames must have \
                 the same size (check the pre-processing settings).",
                file_label(path),
                found.0,
                found.1,
                file_label(reference),
                expected.0,
                expected.1,
            ),
        }
    }
}

impl std::error::Error for FrameError {}

impl FrameSet {
    /// Number of usable frames.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.frames.len()
    }

    /// `true` when no frame survived.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// A short status suffix naming the frames that were left out, if any.
    #[must_use]
    pub fn skipped_note(&self) -> Option<String> {
        let first = self.skipped.first()?;
        let more = self.skipped.len() - 1;
        let tail = if more == 0 {
            String::new()
        } else {
            format!(" and {more} more")
        };
        Some(format!(
            "{} unreadable frame(s) skipped: {}{tail}",
            self.skipped.len(),
            file_label(&first.path)
        ))
    }
}

/// The path list a run works on.
///
/// Temporary result files are removed, the list is optionally reversed
/// (`sort_reverse`), then every `every_nth`-th file is kept (`every_nth <= 1`
/// keeps all). The rule is the shared `stacker_pipeline::order::select_frames_by`,
/// the one the CLI applies too.
#[must_use]
pub fn working_paths(paths: &[PathBuf], sort_reverse: bool, every_nth: u32) -> Vec<PathBuf> {
    select_frames_by(paths.to_vec(), sort_reverse, every_nth)
}

/// `list` with the files of `order` put in that order, each in a position one of
/// them held before: the Sort action's way of rewriting the source list.
///
/// Files that are not in `order` keep their places; files of `order` that are not
/// in `list` are ignored; a repeated file counts once.
#[must_use]
pub fn apply_order(list: &[PathBuf], order: &[PathBuf]) -> Vec<PathBuf> {
    let in_list: std::collections::HashSet<&PathBuf> = list.iter().collect();
    let mut seen = std::collections::HashSet::new();
    let ordered: Vec<&PathBuf> = order
        .iter()
        .filter(|path| in_list.contains(path) && seen.insert(*path))
        .collect();
    let mut next = ordered.iter();
    list.iter()
        .map(|path| {
            if seen.contains(path) {
                next.next()
                    .map_or_else(|| path.clone(), |&ordered| ordered.clone())
            } else {
                path.clone()
            }
        })
        .collect()
}

/// Turn per-file decode outcomes into a [`FrameSet`].
///
/// A failed decode drops the path with the image and is recorded in
/// [`FrameSet::skipped`]. The first decoded frame is the reference; any later
/// frame of a different size is an error naming that file.
///
/// # Errors
///
/// [`FrameError::NoFrames`] when nothing decoded, [`FrameError::DimensionMismatch`]
/// when a frame differs in size from the reference.
pub fn assemble(
    slots: Vec<(PathBuf, Result<PlanarImage<f32>, String>)>,
) -> Result<FrameSet, FrameError> {
    let attempted = slots.len();
    let mut set = FrameSet::default();
    let mut first_reason = None;
    for (path, outcome) in slots {
        match outcome {
            Ok(image) => {
                if let Some(reference) = set.frames.first() {
                    let expected = (reference.width, reference.height);
                    let found = (image.width, image.height);
                    if expected != found {
                        return Err(FrameError::DimensionMismatch {
                            path,
                            reference: set.paths[0].clone(),
                            expected,
                            found,
                        });
                    }
                }
                set.paths.push(path);
                set.frames.push(image);
            }
            Err(reason) => {
                tracing::warn!(path = %path.display(), "failed to load frame: {reason}");
                first_reason.get_or_insert_with(|| reason.clone());
                set.skipped.push(SkippedFrame { path, reason });
            }
        }
    }
    if set.is_empty() {
        return Err(FrameError::NoFrames {
            attempted,
            first_reason,
        });
    }
    Ok(set)
}

/// The decoder's message without the "cannot decode frame '<path>'" prefix: the
/// skipped-frame note names the file itself.
fn decode_reason(error: &PipelineError) -> String {
    match error {
        PipelineError::Decode { message, .. } => message.clone(),
        other => other.to_string(),
    }
}

/// Decode `paths` in parallel with the shared decoder
/// (`stacker_pipeline::load::decode_frame`: EXIF orientation, rotation, crop,
/// resize, linear planar) and assemble the result.
///
/// # Errors
///
/// See [`assemble`].
pub fn decode_frames(
    paths: Vec<PathBuf>,
    pre: &PreprocessingSettings,
) -> Result<FrameSet, FrameError> {
    let slots: Vec<(PathBuf, Result<PlanarImage<f32>, String>)> = paths
        .into_par_iter()
        .map(|path| {
            let outcome = decode_frame(&path, pre).map_err(|e| decode_reason(&e));
            (path, outcome)
        })
        .collect();
    assemble(slots)
}

/// The paths and settings a run was started with, after the working-list rules
/// of the Align and Stack passes have been applied.
///
/// The Align pass (`subsample = false`) aligns every file so that its cache can
/// serve any later `stack_every_nth`; the Stack pass (`subsample = true`) keeps
/// every N-th frame, as the CLI does.
#[must_use]
pub fn paths_for_run(
    paths: &[PathBuf],
    settings: &StackingSettings,
    subsample: bool,
) -> Vec<PathBuf> {
    let nth = if subsample {
        settings.stack_every_nth
    } else {
        1
    };
    working_paths(paths, settings.preprocessing.sort_reverse, nth)
}
