//! The messages between the page and the worker.
//!
//! Every message is one postcard-encoded value in a transferred `ArrayBuffer`
//! (see [`crate::encode`] and [`crate::decode`]). New variants are appended at
//! the end of an enum so the numbering of the older ones never moves; any other
//! change bumps [`PROTOCOL_VERSION`].

use serde::{Deserialize, Serialize};
use stacker_core::image::PlanarImage;
use stacker_pipeline::{Stage as PipelineStage, crop::Rect as CropRect, session};

/// The protocol version.
///
/// The page sends it in [`ToWorker::Hello`] and the worker
/// announces its own in [`FromWorker::Ready`]; a mismatch means the page and the
/// worker come from different builds (a stale cached worker).
///
/// Version 3 appended [`PassKind::Strata`].
pub const PROTOCOL_VERSION: u32 = 3;

/// One file the page hands to the worker: its name and its encoded contents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedBytes {
    /// The file name (the virtual path of the page without its `web:/` prefix).
    pub name: String,
    /// The file as it was read: JPEG, PNG or TIFF.
    pub bytes: Vec<u8>,
}

/// A fusion pass. Mirrors [`session::PassKind`] with a wire format of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PassKind {
    /// Laplacian-pyramid maximum contrast.
    Apex,
    /// Depth map.
    Relief,
    /// Guided-filter fusion (appended in protocol version 3).
    Strata,
}

impl From<session::PassKind> for PassKind {
    fn from(pass: session::PassKind) -> Self {
        match pass {
            session::PassKind::Apex => Self::Apex,
            session::PassKind::Relief => Self::Relief,
            session::PassKind::Strata => Self::Strata,
        }
    }
}

impl From<PassKind> for session::PassKind {
    fn from(pass: PassKind) -> Self {
        match pass {
            PassKind::Apex => Self::Apex,
            PassKind::Relief => Self::Relief,
            PassKind::Strata => Self::Strata,
        }
    }
}

/// A coarse stage of a run. Mirrors [`stacker_pipeline::Stage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stage {
    /// Decoding the frames.
    Load,
    /// Aligning the frames to the reference.
    Align,
    /// The focus census and the auto-cull.
    Cull,
    /// Fusing the frames.
    Fuse,
    /// Encoding and writing the result.
    Save,
}

impl From<PipelineStage> for Stage {
    fn from(stage: PipelineStage) -> Self {
        match stage {
            PipelineStage::Load => Self::Load,
            PipelineStage::Align => Self::Align,
            PipelineStage::Cull => Self::Cull,
            PipelineStage::Fuse => Self::Fuse,
            PipelineStage::Save => Self::Save,
        }
    }
}

impl From<Stage> for PipelineStage {
    fn from(stage: Stage) -> Self {
        match stage {
            Stage::Load => Self::Load,
            Stage::Align => Self::Align,
            Stage::Cull => Self::Cull,
            Stage::Fuse => Self::Fuse,
            Stage::Save => Self::Save,
        }
    }
}

/// The rectangle valid in every aligned frame (the crop). Mirrors
/// [`stacker_pipeline::crop::Rect`], the `(x, y, width, height)` tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    /// Left edge, in pixels.
    pub x: u32,
    /// Top edge, in pixels.
    pub y: u32,
    /// Width, in pixels.
    pub width: u32,
    /// Height, in pixels.
    pub height: u32,
}

/// A pixel count as a wire value (saturating; no image is that large).
fn wire(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

impl Rect {
    /// The wire form of a pipeline crop rectangle.
    #[must_use]
    pub fn from_crop((x, y, width, height): CropRect) -> Self {
        Self {
            x: wire(x),
            y: wire(y),
            width: wire(width),
            height: wire(height),
        }
    }

    /// The pipeline crop rectangle `(x, y, width, height)`.
    #[must_use]
    pub const fn into_crop(self) -> CropRect {
        (
            self.x as usize,
            self.y as usize,
            self.width as usize,
            self.height as usize,
        )
    }
}

/// A planar `f32` image on the wire: the three planes of a
/// [`PlanarImage<f32>`], linear light.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanarF32 {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// First plane (`Y`).
    pub luma: Vec<f32>,
    /// Second plane (`Cb`).
    pub chroma_a: Vec<f32>,
    /// Third plane (`Cr`).
    pub chroma_b: Vec<f32>,
}

impl From<PlanarImage<f32>> for PlanarF32 {
    fn from(image: PlanarImage<f32>) -> Self {
        Self {
            width: wire(image.width),
            height: wire(image.height),
            luma: image.luma,
            chroma_a: image.chroma_a,
            chroma_b: image.chroma_b,
        }
    }
}

impl PlanarF32 {
    /// Back to the pipeline's image. A message whose planes do not match its
    /// size (it cannot be made by [`From`]) becomes an empty image.
    #[must_use]
    pub fn into_planar(self) -> PlanarImage<f32> {
        PlanarImage::from_planes(
            self.width as usize,
            self.height as usize,
            self.luma,
            self.chroma_a,
            self.chroma_b,
        )
        .unwrap_or_else(|_| PlanarImage::new(0, 0))
    }
}

/// A message from the page to the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToWorker {
    /// The first message: the page's [`PROTOCOL_VERSION`].
    Hello {
        /// The page's protocol version.
        version: u32,
    },
    /// Stack the frames: load, align, cull, then one fusion per pass.
    Run {
        /// Identifies the run in every reply.
        job_id: u64,
        /// The effective `StackingSettings` as TOML.
        settings_toml: String,
        /// The fusion passes, in order.
        passes: Vec<PassKind>,
        /// The files in frame order. Empty when `reuse_aligned` and the worker
        /// still has the aligned frames.
        frames: Vec<NamedBytes>,
        /// Use the frames of the last Align (or Run) instead of decoding.
        reuse_aligned: bool,
        /// Pre-resize percentage for this run only (the memory budget's
        /// reduction); `None` keeps the setting.
        pre_resize_override: Option<u32>,
    },
    /// The Align button: load and align, and keep the aligned frames.
    Align {
        /// Identifies the run in every reply.
        job_id: u64,
        /// The effective `StackingSettings` as TOML.
        settings_toml: String,
        /// The files in frame order.
        frames: Vec<NamedBytes>,
        /// Pre-resize percentage for this run only; `None` keeps the setting.
        pre_resize_override: Option<u32>,
    },
    /// The answer to [`FromWorker::ReliefPreview`].
    ReliefThreshold {
        /// The run that asked.
        job_id: u64,
        /// The absolute contrast threshold; `None` cancels the run.
        threshold: Option<f32>,
    },
    /// The aligned, cropped frame called `name` (for retouching).
    FetchSource {
        /// Identifies the request in the reply.
        request_id: u64,
        /// The frame's file name.
        name: String,
    },
    /// Drop the aligned frames and the file contents.
    ForgetAligned,
    /// An sRGB 8-bit thumbnail of the file called `name`, longest edge at most
    /// `max_edge` (the source-file viewer).
    FetchPreview {
        /// Identifies the request in the reply.
        request_id: u64,
        /// The frame's file name.
        name: String,
        /// The longest edge allowed, in pixels.
        max_edge: u32,
    },
}

/// A message from the worker to the page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FromWorker {
    /// The worker is running; its [`PROTOCOL_VERSION`].
    Ready {
        /// The worker's protocol version.
        version: u32,
        /// The WebGPU adapter the worker opened (alignment runs on it); `None`
        /// when this browser or worker offers no WebGPU.
        gpu_adapter: Option<String>,
    },
    /// How far a stage is.
    Progress {
        /// The run.
        job_id: u64,
        /// The stage.
        stage: Stage,
        /// `0.0..=1.0`.
        fraction: f32,
        /// The status text; empty when there is nothing new.
        label: String,
    },
    /// A note for the user (skipped frames, alignment fallbacks, auto-cull).
    Warning {
        /// The run.
        job_id: u64,
        /// The text.
        text: String,
    },
    /// One aligned frame, for the live canvas preview.
    AlignedFrame {
        /// The run.
        job_id: u64,
        /// Frame number.
        index: u32,
        /// Number of frames.
        total: u32,
        /// RGBA, 8 bit, `width * height * 4` bytes.
        preview_rgba: Vec<u8>,
        /// Preview width in pixels.
        width: u32,
        /// Preview height in pixels.
        height: u32,
    },
    /// The `Relief` contrast preview data; the run waits for
    /// [`ToWorker::ReliefThreshold`].
    ReliefPreview {
        /// The run.
        job_id: u64,
        /// Width of `focus_max` and `base_rgba` in pixels.
        width: u32,
        /// Height of `focus_max` and `base_rgba` in pixels.
        height: u32,
        /// The maximum focus map at panel size, row-major.
        focus_max: Vec<f32>,
        /// The (aligned) first frame at panel size, RGBA 8 bit.
        base_rgba: Vec<u8>,
        /// The automatically detected rejected fraction of the full-size map
        /// (`0.0..=1.0`).
        auto_pct: f32,
        /// The absolute threshold of the full-size map for the rejected
        /// fraction `k / (len - 1)` at index `k` (`len` is
        /// [`crate::THRESHOLD_TABLE_LEN`]): the slider position to a threshold.
        thresholds: Vec<f32>,
    },
    /// A finished pass.
    Result {
        /// The run.
        job_id: u64,
        /// Which pass this is.
        pass: PassKind,
        /// The fused image, already cropped to the common valid area.
        planes: PlanarF32,
        /// The names of the frames the auto-cull dropped (first pass only).
        culled: Vec<String>,
        /// Frames that entered the fusion.
        kept: usize,
        /// Frames that decoded.
        total: usize,
        /// A status note: skipped frames, whether the Align cache was reused.
        note: Option<String>,
    },
    /// The Align run finished; the aligned frames are cached.
    AlignDone {
        /// The run.
        job_id: u64,
        /// The common valid area, `None` when no crop is needed.
        crop: Option<Rect>,
        /// Frames that fell back, as one text.
        warning: Option<String>,
        /// The names of the aligned frames, in frame order.
        names: Vec<String>,
    },
    /// The answer to [`ToWorker::FetchSource`].
    Source {
        /// The request.
        request_id: u64,
        /// The frame's file name.
        name: String,
        /// The aligned, cropped frame.
        planes: PlanarF32,
    },
    /// [`ToWorker::FetchSource`] failed.
    SourceFailed {
        /// The request.
        request_id: u64,
        /// Why.
        message: String,
    },
    /// The run failed.
    Failed {
        /// The run (`0` for a failure that belongs to no run).
        job_id: u64,
        /// Why.
        message: String,
    },
    /// The run ended: every pass is published.
    Done {
        /// The run.
        job_id: u64,
    },
    /// The answer to [`ToWorker::FetchPreview`].
    Preview {
        /// The request.
        request_id: u64,
        /// The frame's file name.
        name: String,
        /// Thumbnail width in pixels.
        width: u32,
        /// Thumbnail height in pixels.
        height: u32,
        /// RGBA, 8 bit.
        rgba: Vec<u8>,
    },
    /// The run ended at the `Relief` preview because the user cancelled.
    Cancelled {
        /// The run.
        job_id: u64,
    },
    /// [`ToWorker::FetchPreview`] failed.
    PreviewFailed {
        /// The request.
        request_id: u64,
        /// Why.
        message: String,
    },
}
