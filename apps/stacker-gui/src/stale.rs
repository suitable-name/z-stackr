//! Generation stamps for derived results: which inputs an aligned cache or a
//! stacked result was built from, and whether they still hold.
//!
//! A [`Fingerprint`] has two halves, the *sources* (the ordered file list with each
//! file's size and modified time) and the *settings* the result depends on. A
//! result carries the fingerprint of the inputs it was built from; comparing it
//! with the fingerprint of the inputs *now* ([`stale_reason`]) tells whether the
//! result is still current and, if not, which half moved.
//!
//! The hashing is FNV-1a over a fixed byte layout, so it is deterministic and
//! cheap, and has no dependence on the standard library's hasher. Everything but
//! reading a file's stamp ([`SourceStamp::read`], through the platform seam) is
//! pure.
//!
//! * [`fingerprint_alignment`] hashes the *alignment subset* of the settings: the
//!   alignment mode, the optimizer, the bounded-search, seeding, brightness,
//!   chain and neural-refinement switches, the crop switch (it decides the
//!   border of the warp), the neural model name ([`fingerprint_alignment_with`])
//!   and the whole pre-processing block (it changes the decoded pixels and the
//!   file order). An unrelated setting (JPEG quality, the pyramid depth,
//!   auto-cull, the fusion settings) does not move it. A new alignment-relevant
//!   setting must be added to that function.
//! * [`fingerprint_stack`] hashes every setting that reaches the fused image,
//!   that is the whole [`StackingSettings`] except the `image_saving` block, which
//!   only shapes the file Save writes. It formats the settings with `Debug`, so a
//!   setting added later is covered without touching this module.
//!   [`fingerprint_stack_full`] includes `image_saving` too.

use std::path::{Path, PathBuf};

use crate::{
    platform,
    settings::{AlignmentModeSetting, ComputeDevice, ImageSavingSettings, StackingSettings},
    state::AppState,
};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A streaming FNV-1a hash with length-prefixed strings, so `["ab", "c"]` and
/// `["a", "bc"]` hash differently.
struct Hasher(u64);

impl Hasher {
    fn new(domain: &str) -> Self {
        let mut hasher = Self(FNV_OFFSET);
        hasher.str(domain);
        hasher
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

    fn u128(&mut self, value: u128) {
        self.bytes(&value.to_le_bytes());
    }

    fn bool(&mut self, value: bool) {
        self.bytes(&[u8::from(value)]);
    }

    fn str(&mut self, text: &str) {
        self.u64(u64::try_from(text.len()).unwrap_or(u64::MAX));
        self.bytes(text.as_bytes());
    }

    const fn finish(&self) -> u64 {
        self.0
    }
}

/// The inputs a result was built from, as two hashes. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fingerprint {
    sources: u64,
    settings: u64,
}

/// Which half of the inputs moved since a result was built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaleReason {
    /// The file list changed (added, removed, reordered, or a file was modified).
    Files,
    /// A setting the result depends on changed.
    Settings,
    /// Both changed.
    Both,
}

impl StaleReason {
    /// The badge text: `Stale (files)`, `Stale (settings)`, `Stale (files, settings)`.
    #[must_use]
    pub const fn badge_text(self) -> &'static str {
        match self {
            Self::Files => "Stale (files)",
            Self::Settings => "Stale (settings)",
            Self::Both => "Stale (files, settings)",
        }
    }

    /// What moved, as a phrase for a status line.
    #[must_use]
    pub const fn words(self) -> &'static str {
        match self {
            Self::Files => "the files changed",
            Self::Settings => "the settings changed",
            Self::Both => "the files and the settings changed",
        }
    }
}

/// `None` when `built` and `now` agree, else which half differs.
#[must_use]
pub const fn stale_reason(built: &Fingerprint, now: &Fingerprint) -> Option<StaleReason> {
    match (built.sources != now.sources, built.settings != now.settings) {
        (false, false) => None,
        (true, false) => Some(StaleReason::Files),
        (false, true) => Some(StaleReason::Settings),
        (true, true) => Some(StaleReason::Both),
    }
}

/// One source file as the fingerprint sees it: its path, its size and its
/// modified time (`None` when the file cannot be read, e.g. it was deleted).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceStamp {
    /// The file.
    pub path: PathBuf,
    /// Its size in bytes.
    pub size: Option<u64>,
    /// Its modified time, in nanoseconds since the Unix epoch.
    pub modified_ns: Option<u128>,
}

impl SourceStamp {
    /// Stamp `path` from what the installed [`Platform`](crate::platform::Platform)
    /// knows about the file (its file-system metadata on the desktop).
    #[must_use]
    pub fn read(path: &Path) -> Self {
        let stamp = platform::current().file_stamp(path);
        Self {
            path: path.to_path_buf(),
            size: stamp.map(|s| s.size),
            modified_ns: stamp.and_then(|s| s.modified_ns),
        }
    }
}

/// Stamp every path, in order.
#[must_use]
pub fn source_stamps(paths: &[PathBuf]) -> Vec<SourceStamp> {
    paths.iter().map(|path| SourceStamp::read(path)).collect()
}

/// The sources half of a fingerprint: the ordered stamps, so adding, removing,
/// reordering or modifying a file all change it.
#[must_use]
pub fn fingerprint_sources(stamps: &[SourceStamp]) -> Fingerprint {
    let mut hasher = Hasher::new("sources");
    hasher.u64(u64::try_from(stamps.len()).unwrap_or(u64::MAX));
    for stamp in stamps {
        hasher.str(&stamp.path.to_string_lossy());
        for value in [stamp.size.map(u128::from), stamp.modified_ns] {
            hasher.bool(value.is_some());
            hasher.u128(value.unwrap_or(0));
        }
    }
    Fingerprint {
        sources: hasher.finish(),
        settings: 0,
    }
}

/// [`fingerprint_sources`] over the files' current metadata.
#[must_use]
pub fn fingerprint_paths(paths: &[PathBuf]) -> Fingerprint {
    fingerprint_sources(&source_stamps(paths))
}

/// The fingerprint of an aligned-frame cache built from `sources` with
/// `settings`: the alignment subset only (see the module docs).
#[must_use]
pub fn fingerprint_alignment(sources: &Fingerprint, settings: &StackingSettings) -> Fingerprint {
    fingerprint_alignment_with(sources, settings, "")
}

/// [`fingerprint_alignment`] for a session that picked a neural alignment model.
///
/// `align_model` is a name, not a setting. It counts only in the neural mode, so
/// changing the pick under another mode leaves the cache current.
#[must_use]
pub fn fingerprint_alignment_with(
    sources: &Fingerprint,
    settings: &StackingSettings,
    align_model: &str,
) -> Fingerprint {
    let mut hasher = Hasher::new("alignment");
    hasher.str(settings.alignment_mode.as_combo_str());
    hasher.bool(settings.bounded_search);
    hasher.bool(settings.akaze_seeding);
    hasher.str(&format!("{:?}", settings.optimizer));
    hasher.bool(settings.correct_brightness);
    hasher.bool(settings.chain_alignment);
    hasher.bool(settings.neural_refine_classically);
    // The crop decision and the border handling of the warp.
    hasher.bool(settings.crop_to_common_area);
    if settings.alignment_mode == AlignmentModeSetting::Neural {
        hasher.str(align_model);
    }
    hasher.str(&format!("{:?}", settings.preprocessing));
    Fingerprint {
        sources: sources.sources,
        settings: hasher.finish(),
    }
}

fn stack_hash(domain: &str, settings: &StackingSettings, with_saving: bool) -> u64 {
    let mut hasher = Hasher::new(domain);
    if with_saving {
        hasher.str(&format!("{settings:?}"));
    } else {
        let mut shaping = settings.clone();
        shaping.image_saving = ImageSavingSettings::default();
        // The CPU and the GPU fuse to the same image within a small tolerance, so
        // changing the device does not make a finished result out of date.
        shaping.compute_device = ComputeDevice::default();
        hasher.str(&format!("{shaping:?}"));
    }
    hasher.finish()
}

/// The fingerprint of a stacked result built from `sources` with `settings`:
/// every setting that reaches the fused image (everything but `image_saving`).
#[must_use]
pub fn fingerprint_stack(sources: &Fingerprint, settings: &StackingSettings) -> Fingerprint {
    Fingerprint {
        sources: sources.sources,
        settings: stack_hash("stack", settings, false),
    }
}

/// [`fingerprint_stack`] including the `image_saving` block as well.
#[must_use]
pub fn fingerprint_stack_full(sources: &Fingerprint, settings: &StackingSettings) -> Fingerprint {
    Fingerprint {
        sources: sources.sources,
        settings: stack_hash("stack-full", settings, true),
    }
}

/// A one-line description of the settings a result was built with.
#[must_use]
pub fn summarize_settings(settings: &StackingSettings) -> String {
    let nth = if settings.stack_every_nth > 1 {
        format!(", every {}th frame", settings.stack_every_nth)
    } else {
        String::new()
    };
    format!(
        "{} alignment, auto-cull {}{nth}, {} pyramid levels",
        settings.alignment_mode.as_combo_str(),
        if settings.auto_cull { "on" } else { "off" },
        settings.pyramid_levels,
    )
}

/// What the sidebar shows about currency, derived from the state (the pure half
/// of the `FilesModel` push).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StaleView {
    /// `Aligned frames (N)`, empty when there is no aligned cache.
    pub aligned_summary: String,
    /// Why the aligned cache is out of date, `None` when current or absent.
    pub aligned_stale: Option<StaleReason>,
    /// Why the newest result is out of date, `None` when current or absent.
    pub result_stale: Option<StaleReason>,
    /// One flag per source file: dropped by the last auto-cull.
    pub culled: Vec<bool>,
}

/// Derive the [`StaleView`] of `state` against `settings`. The file metadata is
/// read only when there is an aligned cache or a result to compare.
#[must_use]
pub fn stale_view(state: &AppState, settings: &StackingSettings) -> StaleView {
    let needs_sources = state.aligned.is_some() || !state.result_paths.is_empty();
    let sources = needs_sources.then(|| fingerprint_paths(&state.file_paths));
    let (aligned_stale, result_stale) = sources.as_ref().map_or((None, None), |sources| {
        (
            state.aligned_staleness(sources, settings),
            state
                .result_paths
                .len()
                .checked_sub(1)
                .and_then(|newest| state.result_staleness(newest, sources, settings)),
        )
    });
    StaleView {
        aligned_summary: state.aligned.as_ref().map_or_else(String::new, |cache| {
            format!("Aligned frames ({})", cache.paths.len())
        }),
        aligned_stale,
        result_stale,
        culled: state.culled_flags(),
    }
}
