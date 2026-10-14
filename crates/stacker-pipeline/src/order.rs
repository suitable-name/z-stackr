//! Which files are frames, and in which order they enter the stack.
//!
//! * [`collect_image_paths`] scans a directory (non-recursive, common image
//!   extensions) and returns the frames in **natural** file-name order
//!   (`frame_2` before `frame_10`), without the pipeline's own products.
//! * [`select_frames`] applies the settings that shape the frame list:
//!   temporary result files are dropped, `preprocessing.sort_reverse` reverses
//!   the order (so the alignment reference is the other end of the stack), and
//!   `stack_every_nth` keeps every N-th frame of the result. The first frame of
//!   the resulting list is the alignment reference.
//!
//! Both apps call these, so the same folder and settings give the same stack.

use std::{
    cmp::Ordering,
    iter::Peekable,
    path::{Path, PathBuf},
    str::Chars,
};

use stacker_core::settings::StackingSettings;

use crate::PipelineError;

/// File-name marker of the desktop app's temporary result images; they are
/// products of a run and never source frames.
pub const RESULT_MARKER: &str = "stacker_result_";

/// The (lower-case, dot-less) extensions the pipeline reads as frames. The
/// single list: [`has_frame_extension`], the directory scan and the apps'
/// file-picker filters all use it.
pub const FRAME_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "tif", "tiff"];

/// `true` when `path` has one of the [`FRAME_EXTENSIONS`], in any letter case.
///
/// Only the extension is checked; callers that need a regular file test that
/// themselves (the monitor loop gets paths of files that may already be gone).
#[must_use]
pub fn has_frame_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| {
            FRAME_EXTENSIONS
                .iter()
                .chain(crate::load::RAW_FRAME_EXTENSIONS)
                .any(|known| known.eq_ignore_ascii_case(ext))
        })
}

/// `true` for the temporary result images of the desktop app.
#[must_use]
pub fn is_result_file(path: &Path) -> bool {
    path.file_name()
        .is_some_and(|n| n.to_string_lossy().contains(RESULT_MARKER))
}

/// Resolve `path` to a comparable form even when the file does not exist yet:
/// canonicalise the file itself, else its parent directory plus the file name.
#[must_use]
pub fn normalize_path(path: &Path) -> PathBuf {
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return canonical;
    }
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        if let Ok(canonical_parent) = std::fs::canonicalize(parent) {
            return canonical_parent.join(name);
        }
    }
    path.to_path_buf()
}

/// `true` when `path` names the same file as one of `exclude`.
///
/// The exclusion list holds the pipeline's own products (output image, log
/// file); without it a run whose output sits in the input directory would pick
/// its previous result up as an extra frame.
#[must_use]
pub fn is_excluded(path: &Path, exclude: &[PathBuf]) -> bool {
    if exclude.is_empty() {
        return false;
    }
    let candidate = normalize_path(path);
    exclude.iter().any(|e| normalize_path(e) == candidate)
}

/// Compare two file names so that digit runs are ordered by numeric value
/// (`frame_2` before `frame_10`) and everything else case-insensitively.
///
/// Ties (`frame_01` against `frame_1`, or names differing only in case) compare
/// equal here; callers add a plain string comparison for a total order.
#[must_use]
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    fn take_digits(it: &mut Peekable<Chars<'_>>) -> String {
        let mut digits = String::new();
        while let Some(&c) = it.peek() {
            if !c.is_ascii_digit() {
                break;
            }
            digits.push(c);
            it.next();
        }
        digits
    }

    let mut ai = a.chars().peekable();
    let mut bi = b.chars().peekable();
    loop {
        match (ai.peek().copied(), bi.peek().copied()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(ca), Some(cb)) if ca.is_ascii_digit() && cb.is_ascii_digit() => {
                let da = take_digits(&mut ai);
                let db = take_digits(&mut bi);
                let ta = da.trim_start_matches('0');
                let tb = db.trim_start_matches('0');
                let ord = ta.len().cmp(&tb.len()).then_with(|| ta.cmp(tb));
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            (Some(ca), Some(cb)) => {
                let (la, lb) = (ca.to_ascii_lowercase(), cb.to_ascii_lowercase());
                if la != lb {
                    return la.cmp(&lb);
                }
                ai.next();
                bi.next();
            }
        }
    }
}

/// Sort `paths` by file name in natural order (a plain path comparison breaks
/// ties, so the order is total and reproducible).
pub fn natural_sort(paths: &mut [PathBuf]) {
    let name_of = |p: &Path| {
        p.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    paths.sort_by(|a, b| natural_cmp(&name_of(a), &name_of(b)).then_with(|| a.cmp(b)));
}

/// Sort file `names` in natural order, the rule of [`natural_sort`] for names
/// without a directory: [`natural_cmp`], then a plain string comparison for a
/// total, reproducible order.
pub fn natural_sort_names(names: &mut [String]) {
    names.sort_by(|a, b| natural_cmp(a, b).then_with(|| a.cmp(b)));
}

/// Collect the frame paths of `dir` (non-recursive, common image extensions),
/// in natural file-name order, without any path listed in `exclude` and
/// without the desktop app's temporary result images.
///
/// The first path of the result is the alignment reference (before
/// [`select_frames`] possibly reverses the list).
///
/// # Errors
/// Returns an error when the directory cannot be read, or
/// [`PipelineError::NoImages`] when it holds no usable frame.
pub fn collect_image_paths(dir: &Path, exclude: &[PathBuf]) -> Result<Vec<PathBuf>, PipelineError> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_file() && has_frame_extension(p) && !is_result_file(p) && !is_excluded(p, exclude)
        })
        .collect();
    natural_sort(&mut paths);
    if paths.is_empty() {
        return Err(PipelineError::NoImages(dir.to_path_buf()));
    }
    Ok(paths)
}

/// The frames a run works on, from an already ordered list.
///
/// Temporary result files are removed, the list is optionally reversed
/// (`sort_reverse`), then every `every_nth`-th frame is kept (`every_nth <= 1`
/// keeps all). The first frame of the (reversed) list is always kept, so a
/// non-empty input stays non-empty.
#[must_use]
pub fn select_frames_by(paths: Vec<PathBuf>, sort_reverse: bool, every_nth: u32) -> Vec<PathBuf> {
    select_by(paths, |p| !is_result_file(p), sort_reverse, every_nth)
}

/// [`select_frames_by`] for file names instead of paths: the same rules (names
/// containing [`RESULT_MARKER`] are dropped, `sort_reverse`, every N-th), so a
/// caller without a file system selects the same frames.
#[must_use]
pub fn select_frame_names(names: Vec<String>, sort_reverse: bool, every_nth: u32) -> Vec<String> {
    select_by(
        names,
        |name| !name.contains(RESULT_MARKER),
        sort_reverse,
        every_nth,
    )
}

/// The selection rule behind [`select_frames_by`] and [`select_frame_names`].
fn select_by<T>(
    items: Vec<T>,
    keep: impl Fn(&T) -> bool,
    sort_reverse: bool,
    every_nth: u32,
) -> Vec<T> {
    let mut work: Vec<T> = items.into_iter().filter(|item| keep(item)).collect();
    if sort_reverse {
        work.reverse();
    }
    let nth = every_nth.max(1) as usize;
    if nth > 1 {
        work = work
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % nth == 0)
            .map(|(_, item)| item)
            .collect();
    }
    work
}

/// [`select_frames_by`] with the keys read from the settings
/// (`preprocessing.sort_reverse`, `stack_every_nth`).
#[must_use]
pub fn select_frames(paths: Vec<PathBuf>, settings: &StackingSettings) -> Vec<PathBuf> {
    select_frames_by(
        paths,
        settings.preprocessing.sort_reverse,
        settings.stack_every_nth,
    )
}
