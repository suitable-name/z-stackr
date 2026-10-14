//! Frame preprocessing (rotation, crop, resize).
//!
//! The implementation lives in the shared pipeline so the desktop app and the
//! CLI cannot drift apart; the functions are re-exported here.
//! `stacker_pipeline::load::decode_frame` additionally applies the EXIF
//! orientation before these steps (`ignore_exif_orientation`).

pub use stacker_pipeline::load::{parse_crop_spec, preprocess_frame};
