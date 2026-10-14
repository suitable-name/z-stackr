//! The z-stackr project file: one session as `<name>.zstack.toml`.
//!
//! It holds the source list, the settings and the retouch strokes, and lives
//! next to the source images. The strokes are stored replayable.
//!
//! # File layout
//!
//! ```toml
//! version = 1
//! output_name = "blossom"          # optional
//! sources = ["IMG_0001.jpg", "IMG_0002.jpg"]
//!
//! [settings]                       # the StackingSettings tables
//! # ...
//!
//! [[strokes]]                      # one per brush drag, oldest first
//! source_index = 1                 # position in `sources`; optional
//! radius = 12.0
//! opacity = 1.0
//! points = [10, 20, 11, 20]        # x0, y0, x1, y1, ...
//! ```
//!
//! Source paths are stored relative to the project file (with `/` separators)
//! when they lie below its folder, else absolute; loading turns them into paths
//! that work as they are. A stroke point of `4294967295, 4294967295` is a line
//! break (the pointer left the image).
//!
//! The reader is lenient: unknown keys are ignored, a wrongly typed setting
//! keeps its default (reported as a warning), and a file from a newer version
//! loads with a warning.
//!
//! # Dirty rule
//!
//! A project is *dirty* when its content differs from the content last saved or
//! opened. [`Project::fingerprint`] hashes everything the file stores, so the
//! comparison is by content, never by a flag: saving twice leaves it clean,
//! reordering the sources makes it dirty, and undoing a change cleans it again
//! ([`Baseline`]).

use std::{
    fmt::Write as _,
    io,
    path::{Path, PathBuf},
};

use stacker_pipeline::config::{Leniency, parse_settings};
use thiserror::Error;
use toml::{Table, Value};

use crate::{platform, retouch::Stroke, settings::StackingSettings};

/// The file-format version this build writes.
pub const CURRENT_VERSION: u32 = 1;

/// The ending of a project file name.
pub const FILE_SUFFIX: &str = ".zstack.toml";

/// Why a project could not be read or written.
#[derive(Debug, Error)]
pub enum ProjectError {
    /// Reading or writing the file failed.
    #[error("{0}")]
    Io(#[from] io::Error),
    /// The text is not a project file.
    #[error("not a valid project file: {0}")]
    Parse(String),
    /// The project could not be turned into TOML.
    #[error("the project could not be encoded: {0}")]
    Encode(String),
}

/// One brush drag as the project file stores it (see [`Stroke`] for the live
/// form).
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeRecord {
    /// The position in [`Project::sources`] of the frame painted from.
    pub source_index: Option<usize>,
    /// The raster pixels visited; `(u32::MAX, u32::MAX)` breaks the line.
    pub points: Vec<(u32, u32)>,
    /// Brush radius in raster pixels.
    pub radius: f32,
    /// Brush opacity in `0..=1`.
    pub opacity: f32,
}

impl StrokeRecord {
    /// The record of the live stroke `stroke`, naming its source frame by its
    /// position in `sources`.
    #[must_use]
    pub fn from_stroke(stroke: &Stroke, sources: &[PathBuf]) -> Self {
        Self {
            source_index: stroke
                .source
                .as_ref()
                .and_then(|path| sources.iter().position(|p| p == path)),
            points: stroke.points.clone(),
            radius: stroke.radius,
            opacity: stroke.opacity,
        }
    }

    /// The live stroke this record replays as; the source frame is looked up in
    /// `sources` (`None` when the index is gone).
    #[must_use]
    pub fn to_stroke(&self, sources: &[PathBuf]) -> Stroke {
        Stroke {
            source: self.source_index.and_then(|i| sources.get(i)).cloned(),
            points: self.points.clone(),
            radius: self.radius,
            opacity: self.opacity,
        }
    }
}

/// The content of a project file.
#[derive(Debug, Clone, PartialEq)]
pub struct Project {
    /// The file-format version.
    pub version: u32,
    /// The source images in list order. In memory these are the paths the app
    /// uses; the file stores them relative to itself where it can.
    pub sources: Vec<PathBuf>,
    /// The stacking settings of the session.
    pub settings: StackingSettings,
    /// The result name the export uses, if the user chose one.
    pub output_name: Option<String>,
    /// The retouch strokes, oldest first.
    pub strokes: Vec<StrokeRecord>,
}

/// A loaded project and the notes the reader made.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedProject {
    /// The project.
    pub project: Project,
    /// Ignored keys, defaulted settings and the like; empty for a clean file.
    pub warnings: Vec<String>,
}

impl Project {
    /// A project over `sources` with `settings` and nothing else.
    #[must_use]
    pub const fn new(sources: Vec<PathBuf>, settings: StackingSettings) -> Self {
        Self {
            version: CURRENT_VERSION,
            sources,
            settings,
            output_name: None,
            strokes: Vec::new(),
        }
    }

    /// Hash of everything the file stores (FNV-1a over a fixed layout), for the
    /// dirty rule. Independent of where the file lives.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let mut hasher = Fnv::new();
        hasher.u64(u64::from(self.version));
        hasher.usize(self.sources.len());
        for source in &self.sources {
            hasher.str(&source.to_string_lossy());
        }
        hasher.str(&format!("{:?}", self.settings));
        match &self.output_name {
            Some(name) => {
                hasher.u64(1);
                hasher.str(name);
            }
            None => hasher.u64(0),
        }
        hasher.usize(self.strokes.len());
        for stroke in &self.strokes {
            match stroke.source_index {
                Some(i) => {
                    hasher.u64(1);
                    hasher.usize(i);
                }
                None => hasher.u64(0),
            }
            hasher.u64(u64::from(stroke.radius.to_bits()));
            hasher.u64(u64::from(stroke.opacity.to_bits()));
            hasher.usize(stroke.points.len());
            for &(x, y) in &stroke.points {
                hasher.u64(u64::from(x));
                hasher.u64(u64::from(y));
            }
        }
        hasher.finish()
    }

    /// The project as TOML text. `base` is the folder of the file it will be
    /// written to: sources below it are stored relative to it.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Encode`] when the settings cannot be encoded.
    pub fn to_toml(&self, base: Option<&Path>) -> Result<String, ProjectError> {
        let mut table = Table::new();
        table.insert(
            "version".to_owned(),
            Value::Integer(i64::from(self.version)),
        );
        if let Some(name) = &self.output_name {
            table.insert("output_name".to_owned(), Value::String(name.clone()));
        }
        let sources = self
            .sources
            .iter()
            .map(|path| Value::String(store_path(path, base)))
            .collect();
        table.insert("sources".to_owned(), Value::Array(sources));
        let settings =
            Value::try_from(&self.settings).map_err(|e| ProjectError::Encode(e.to_string()))?;
        table.insert("settings".to_owned(), settings);
        if !self.strokes.is_empty() {
            let strokes = self.strokes.iter().map(stroke_to_value).collect();
            table.insert("strokes".to_owned(), Value::Array(strokes));
        }
        toml::to_string(&table).map_err(|e| ProjectError::Encode(e.to_string()))
    }

    /// Read a project from TOML text. Relative source paths are resolved
    /// against `base` (the folder of the file; `None` leaves them as written).
    ///
    /// # Errors
    ///
    /// [`ProjectError::Parse`] when the text is not TOML or `sources` is not a
    /// list of strings. Everything else is repaired and reported as a warning.
    pub fn from_toml(text: &str, base: Option<&Path>) -> Result<LoadedProject, ProjectError> {
        let table: Table = toml::from_str(text).map_err(|e| ProjectError::Parse(e.to_string()))?;
        let mut warnings = Vec::new();

        let version = match table.get("version") {
            Some(Value::Integer(v)) => u32::try_from(*v).unwrap_or(CURRENT_VERSION),
            Some(_) => {
                warnings.push("`version` is not a number; assuming the current one".to_owned());
                CURRENT_VERSION
            }
            None => CURRENT_VERSION,
        };
        if version > CURRENT_VERSION {
            warnings.push(format!(
                "the file is version {version}, this build reads version {CURRENT_VERSION}; \
                 newer entries are ignored"
            ));
        }

        let sources = match table.get("sources") {
            None => Vec::new(),
            Some(Value::Array(items)) => {
                let mut sources = Vec::with_capacity(items.len());
                for item in items {
                    if let Value::String(path) = item {
                        sources.push(resolve_path(path, base));
                    } else {
                        warnings.push("a `sources` entry is not a string; skipped".to_owned());
                    }
                }
                sources
            }
            Some(_) => {
                return Err(ProjectError::Parse(
                    "`sources` must be a list of file paths".to_owned(),
                ));
            }
        };

        let output_name = match table.get("output_name") {
            Some(Value::String(name)) if !name.is_empty() => Some(name.clone()),
            _ => None,
        };

        let settings = match table.get("settings") {
            Some(Value::Table(settings)) => {
                let text =
                    toml::to_string(settings).map_err(|e| ProjectError::Parse(e.to_string()))?;
                let loaded =
                    parse_settings(&text, Leniency::Lenient).map_err(ProjectError::Parse)?;
                warnings.extend(
                    loaded
                        .warnings
                        .into_iter()
                        .map(|w| format!("settings: {w}")),
                );
                loaded.settings
            }
            Some(_) => {
                warnings.push("`settings` is not a table; defaults are used".to_owned());
                StackingSettings::default()
            }
            None => StackingSettings::default(),
        };

        let mut strokes = Vec::new();
        match table.get("strokes") {
            None => {}
            Some(Value::Array(items)) => {
                for item in items {
                    match stroke_from_value(item) {
                        Some(stroke) => strokes.push(stroke),
                        None => warnings.push("a stroke is malformed; skipped".to_owned()),
                    }
                }
            }
            Some(_) => warnings.push("`strokes` is not a list; ignored".to_owned()),
        }

        Ok(LoadedProject {
            project: Self {
                version,
                sources,
                settings,
                output_name,
                strokes,
            },
            warnings,
        })
    }

    /// Write the project to `path` through the installed platform (atomically on
    /// the desktop, a download in the browser), sources relative to its folder.
    ///
    /// # Errors
    ///
    /// The encoding error, or the I/O error of the write.
    pub fn save(&self, path: &Path) -> Result<(), ProjectError> {
        let text = self.to_toml(path.parent())?;
        platform::current().write_bytes(path, text.as_bytes())?;
        Ok(())
    }

    /// Read the project at `path` through the installed platform; relative
    /// sources resolve against its folder.
    ///
    /// # Errors
    ///
    /// The I/O error of the read, or [`ProjectError::Parse`].
    pub fn load(path: &Path) -> Result<LoadedProject, ProjectError> {
        let text = platform::current().read_to_string(path)?;
        Self::from_toml(&text, path.parent())
    }
}

// ── The dirty rule ───────────────────────────────────────────────────────────

/// The fingerprint of the project content at the last save or open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Baseline(u64);

impl Baseline {
    /// The baseline of `project`.
    #[must_use]
    pub fn of(project: &Project) -> Self {
        Self(project.fingerprint())
    }

    /// `true` when `project` differs in content from the baseline.
    #[must_use]
    pub fn is_dirty(self, project: &Project) -> bool {
        self.0 != project.fingerprint()
    }
}

// ── File names ───────────────────────────────────────────────────────────────

/// The display name of the project file at `path`: its file name without the
/// `.zstack.toml` ending (or without any extension for another name).
#[must_use]
pub fn project_name(path: &Path) -> String {
    let Some(file) = path.file_name().map(|n| n.to_string_lossy().into_owned()) else {
        return path.display().to_string();
    };
    if let Some(stem) = strip_suffix(&file) {
        return stem.to_owned();
    }
    path.file_stem()
        .map_or(file, |stem| stem.to_string_lossy().into_owned())
}

/// `name` without the `.zstack.toml` ending (compared without regard to ASCII
/// case); `None` when it does not end with it.
fn strip_suffix(name: &str) -> Option<&str> {
    let cut = name.len().checked_sub(FILE_SUFFIX.len())?;
    if !name.is_char_boundary(cut) || !name[cut..].eq_ignore_ascii_case(FILE_SUFFIX) {
        return None;
    }
    Some(&name[..cut])
}

/// `path` with the `.zstack.toml` ending: a name without it gets it (any other
/// extension is replaced), so a typed "blossom" becomes `blossom.zstack.toml`.
#[must_use]
pub fn with_project_suffix(path: PathBuf) -> PathBuf {
    let has_suffix = path
        .file_name()
        .is_some_and(|name| strip_suffix(&name.to_string_lossy()).is_some());
    if has_suffix {
        return path;
    }
    let name = project_name(&path);
    path.with_file_name(format!("{name}{FILE_SUFFIX}"))
}

/// Where to offer saving a project of `sources`: next to the first source,
/// named after that folder. `None` for an empty list.
#[must_use]
pub fn suggested_location(sources: &[PathBuf]) -> Option<(PathBuf, String)> {
    let folder = sources.first()?.parent()?.to_path_buf();
    let name = folder
        .file_name()
        .map_or_else(|| "stack".to_owned(), |n| n.to_string_lossy().into_owned());
    Some((folder, format!("{name}{FILE_SUFFIX}")))
}

// ── Paths and strokes in the file ────────────────────────────────────────────

/// `path` as stored: relative to `base` with `/` separators when it lies below
/// it, else as is.
fn store_path(path: &Path, base: Option<&Path>) -> String {
    if let Some(base) = base
        && let Ok(relative) = path.strip_prefix(base)
        && !relative.as_os_str().is_empty()
    {
        let mut text = String::new();
        for (i, part) in relative.components().enumerate() {
            if i > 0 {
                text.push('/');
            }
            text.push_str(&part.as_os_str().to_string_lossy());
        }
        return text;
    }
    path.to_string_lossy().into_owned()
}

/// A stored path as a usable one: absolute stays, relative joins `base`.
fn resolve_path(stored: &str, base: Option<&Path>) -> PathBuf {
    let absolute = Path::new(stored);
    let Some(base) = base else {
        return absolute.to_path_buf();
    };
    if absolute.is_absolute() {
        return absolute.to_path_buf();
    }
    let mut path = base.to_path_buf();
    for part in stored
        .split(['/', '\\'])
        .filter(|p| !p.is_empty() && *p != ".")
    {
        path.push(part);
    }
    path
}

fn stroke_to_value(stroke: &StrokeRecord) -> Value {
    let mut table = Table::new();
    if let Some(index) = stroke.source_index
        && let Ok(index) = i64::try_from(index)
    {
        table.insert("source_index".to_owned(), Value::Integer(index));
    }
    table.insert("radius".to_owned(), Value::Float(f64::from(stroke.radius)));
    table.insert(
        "opacity".to_owned(),
        Value::Float(f64::from(stroke.opacity)),
    );
    let points = stroke
        .points
        .iter()
        .flat_map(|&(x, y)| [Value::Integer(i64::from(x)), Value::Integer(i64::from(y))])
        .collect();
    table.insert("points".to_owned(), Value::Array(points));
    Value::Table(table)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "brush sizes are tiny; f32 is the live precision"
)]
fn number(value: &Value) -> Option<f32> {
    match value {
        Value::Float(f) => Some(*f as f32),
        Value::Integer(i) => i32::try_from(*i).ok().map(|i| f64::from(i) as f32),
        _ => None,
    }
}

fn stroke_from_value(value: &Value) -> Option<StrokeRecord> {
    let table = value.as_table()?;
    let radius = number(table.get("radius")?)?;
    let opacity = table.get("opacity").and_then(number).unwrap_or(1.0);
    let source_index = match table.get("source_index") {
        Some(Value::Integer(i)) => usize::try_from(*i).ok(),
        _ => None,
    };
    let (pairs, rest) = table.get("points")?.as_array()?.as_slice().as_chunks::<2>();
    if !rest.is_empty() {
        return None;
    }
    let coordinate = |value: &Value| match value {
        Value::Integer(i) => u32::try_from(*i).ok(),
        _ => None,
    };
    let mut points = Vec::with_capacity(pairs.len());
    for [x, y] in pairs {
        points.push((coordinate(x)?, coordinate(y)?));
    }
    Some(StrokeRecord {
        source_index,
        points,
        radius,
        opacity,
    })
}

/// A one-line description of a project for logs and tooltips.
#[must_use]
pub fn summary(project: &Project) -> String {
    let mut text = format!("{} image(s)", project.sources.len());
    if !project.strokes.is_empty() {
        let _ = write!(text, ", {} retouch stroke(s)", project.strokes.len());
    }
    text
}

// ── Hashing ──────────────────────────────────────────────────────────────────

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A streaming FNV-1a hash with length-prefixed strings.
struct Fnv(u64);

impl Fnv {
    const fn new() -> Self {
        Self(FNV_OFFSET)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(u64::try_from(value).unwrap_or(u64::MAX));
    }

    fn str(&mut self, text: &str) {
        self.usize(text.len());
        self.bytes(text.as_bytes());
    }

    const fn finish(&self) -> u64 {
        self.0
    }
}
