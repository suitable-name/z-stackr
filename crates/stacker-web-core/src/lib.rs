//! The contract between the z-stackr browser page and its Web Worker.
//!
//! The page (`apps/stacker-web`) runs the desktop's Slint interface; the worker
//! (`apps/stacker-web-worker`) runs the stacking pipeline. They talk through the
//! messages of [`protocol`], each one postcard-encoded ([`encode`], [`decode`]) in
//! a transferred `ArrayBuffer`. This crate also holds the policy both sides share:
//! the worker's memory [`budget`] and the small-picture helpers of [`thumb`].
//!
//! Nothing here touches the browser, so all of it is tested natively.

#![warn(missing_docs)]

pub mod budget;
pub mod protocol;
pub mod thumb;

use serde::{Serialize, de::DeserializeOwned};

pub use budget::{Budget, MARGIN_BYTES, WORKER_BUDGET_BYTES, dims_from_bytes, plan_budget};
pub use protocol::{
    FromWorker, NamedBytes, PROTOCOL_VERSION, PassKind, PlanarF32, Rect, Stage, ToWorker,
};
pub use stacker_pipeline::session::{THRESHOLD_TABLE_LEN, threshold_table};
pub use thumb::{Rgba, downsample_plane, planar_to_rgba, thumbnail_size};

/// The bytes of a message.
///
/// Serialising these plain data types cannot fail; the
/// empty vector returned in that impossible case fails to [`decode`] on the other
/// side with a clear error instead of panicking here.
#[must_use]
pub fn encode<T: Serialize>(message: &T) -> Vec<u8> {
    postcard::to_allocvec(message).unwrap_or_default()
}

/// A message from its bytes.
///
/// # Errors
/// The decoder's message when the bytes are not a `T` (a truncated message, or a
/// page and a worker built from different protocol versions).
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    postcard::from_bytes(bytes).map_err(|error| format!("cannot decode a message: {error}"))
}

#[cfg(test)]
mod tests {
    use stacker_core::image::PlanarImage;

    use super::{
        FromWorker, NamedBytes, PROTOCOL_VERSION, PassKind, PlanarF32, Rect, Stage, ToWorker,
        decode, encode,
    };

    fn round_trip<T>(message: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let back: T = decode(&encode(message)).expect("a message decodes");
        assert_eq!(&back, message);
    }

    fn planes() -> PlanarF32 {
        let mut image = PlanarImage::new(3, 2);
        image.luma = vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.125];
        image.chroma_a[1] = -0.25;
        image.chroma_b[4] = 0.5;
        image.into()
    }

    #[test]
    fn the_version_is_three() {
        assert_eq!(PROTOCOL_VERSION, 3);
    }

    #[test]
    fn the_strata_pass_is_appended_and_maps_to_the_session_pass() {
        use stacker_pipeline::session::PassKind as SessionPass;

        // Append-only numbering: the older variants keep their postcard index.
        assert_eq!(encode(&PassKind::Apex), [0]);
        assert_eq!(encode(&PassKind::Relief), [1]);
        assert_eq!(encode(&PassKind::Strata), [2]);
        for pass in [PassKind::Apex, PassKind::Relief, PassKind::Strata] {
            assert_eq!(PassKind::from(SessionPass::from(pass)), pass);
        }
    }

    #[test]
    fn every_message_to_the_worker_round_trips() {
        let frames = vec![NamedBytes {
            name: "a.jpg".to_owned(),
            bytes: vec![1, 2, 3],
        }];
        round_trip(&ToWorker::Hello {
            version: PROTOCOL_VERSION,
        });
        round_trip(&ToWorker::Run {
            job_id: 7,
            settings_toml: "alignment_mode = \"none\"".to_owned(),
            passes: vec![PassKind::Relief, PassKind::Apex, PassKind::Strata],
            frames: frames.clone(),
            reuse_aligned: false,
            pre_resize_override: Some(65),
        });
        round_trip(&ToWorker::Align {
            job_id: 8,
            settings_toml: String::new(),
            frames,
            pre_resize_override: None,
        });
        round_trip(&ToWorker::ReliefThreshold {
            job_id: 7,
            threshold: Some(0.125),
        });
        round_trip(&ToWorker::ReliefThreshold {
            job_id: 7,
            threshold: None,
        });
        round_trip(&ToWorker::FetchSource {
            request_id: 3,
            name: "a.jpg".to_owned(),
        });
        round_trip(&ToWorker::ForgetAligned);
        round_trip(&ToWorker::FetchPreview {
            request_id: 4,
            name: "a.jpg".to_owned(),
            max_edge: 256,
        });
    }

    #[test]
    fn every_message_from_the_worker_round_trips() {
        round_trip(&FromWorker::Ready {
            version: PROTOCOL_VERSION,
            gpu_adapter: Some("NVIDIA GeForce RTX 3080".to_owned()),
        });
        round_trip(&FromWorker::Ready {
            version: PROTOCOL_VERSION,
            gpu_adapter: None,
        });
        round_trip(&FromWorker::Progress {
            job_id: 1,
            stage: Stage::Align,
            fraction: 0.5,
            label: "Aligning".to_owned(),
        });
        round_trip(&FromWorker::Warning {
            job_id: 1,
            text: "skipped".to_owned(),
        });
        round_trip(&FromWorker::AlignedFrame {
            job_id: 1,
            index: 2,
            total: 9,
            preview_rgba: vec![0, 1, 2, 255],
            width: 1,
            height: 1,
        });
        round_trip(&FromWorker::ReliefPreview {
            job_id: 1,
            width: 2,
            height: 1,
            focus_max: vec![0.5, 1.5],
            base_rgba: vec![0; 8],
            auto_pct: 0.4,
            thresholds: vec![0.0, 0.5, 1.0],
        });
        round_trip(&FromWorker::Result {
            job_id: 1,
            pass: PassKind::Apex,
            planes: planes(),
            culled: vec!["b.jpg".to_owned()],
            kept: 3,
            total: 4,
            note: Some("reused".to_owned()),
        });
        round_trip(&FromWorker::Result {
            job_id: 1,
            pass: PassKind::Strata,
            planes: planes(),
            culled: Vec::new(),
            kept: 4,
            total: 4,
            note: None,
        });
        round_trip(&FromWorker::AlignDone {
            job_id: 1,
            crop: Some(Rect::from_crop((1, 2, 30, 40))),
            warning: None,
            names: vec!["a.jpg".to_owned(), "b.jpg".to_owned()],
        });
        round_trip(&FromWorker::Source {
            request_id: 5,
            name: "a.jpg".to_owned(),
            planes: planes(),
        });
        round_trip(&FromWorker::SourceFailed {
            request_id: 5,
            message: "no".to_owned(),
        });
        round_trip(&FromWorker::Failed {
            job_id: 1,
            message: "boom".to_owned(),
        });
        round_trip(&FromWorker::Done { job_id: 1 });
        round_trip(&FromWorker::Preview {
            request_id: 6,
            name: "a.jpg".to_owned(),
            width: 1,
            height: 1,
            rgba: vec![9, 8, 7, 255],
        });
        round_trip(&FromWorker::Cancelled { job_id: 1 });
        round_trip(&FromWorker::PreviewFailed {
            request_id: 6,
            message: "no".to_owned(),
        });
    }

    #[test]
    fn planes_survive_the_wire_unchanged() {
        let wire = planes();
        let image = wire.clone().into_planar();
        assert_eq!((image.width, image.height), (3, 2));
        assert!(image.is_consistent());
        assert_eq!(PlanarF32::from(image), wire);
    }

    #[test]
    fn the_crop_rectangle_round_trips() {
        let crop = (4, 5, 600, 400);
        assert_eq!(Rect::from_crop(crop).into_crop(), crop);
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(decode::<ToWorker>(&[0xff, 0xff, 0xff]).is_err());
        assert!(decode::<FromWorker>(&[]).is_err());
    }
}
