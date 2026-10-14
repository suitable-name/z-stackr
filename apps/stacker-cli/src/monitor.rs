//! Decisions of the `--monitor` loop that can be tested without a file watcher.
use crate::pipeline::{has_frame_extension, is_excluded};
use notify::{Event, EventKind};
use std::path::PathBuf;

/// `true` when a watcher event should start a new stacking pass.
///
/// A file must have been created or modified, and at least one affected path
/// must be an input frame that is not one of the pipeline's own products
/// (`exclude`: output image, log file).  Without the exclusion a result written
/// into the watched directory would retrigger the run that wrote it.
#[must_use]
pub fn is_relevant_event(event: &Event, exclude: &[PathBuf]) -> bool {
    matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_))
        && event
            .paths
            .iter()
            .any(|p| has_frame_extension(p) && !is_excluded(p, exclude))
}

#[cfg(test)]
mod tests {
    use super::*;
    use notify::event::{CreateKind, ModifyKind, RemoveKind};

    fn event(kind: EventKind, path: &str) -> Event {
        Event::new(kind).add_path(PathBuf::from(path))
    }

    #[test]
    fn new_frames_trigger_a_run() {
        let create = event(EventKind::Create(CreateKind::File), "watched/frame_11.png");
        assert!(is_relevant_event(&create, &[]));
        let modify = event(EventKind::Modify(ModifyKind::Any), "watched/frame_11.JPG");
        assert!(is_relevant_event(&modify, &[]));
    }

    #[test]
    fn own_output_non_images_and_removals_are_ignored() {
        let out = PathBuf::from("watched/result.png");
        let exclude = std::slice::from_ref(&out);

        let own = event(EventKind::Modify(ModifyKind::Any), "watched/result.png");
        assert!(!is_relevant_event(&own, exclude));

        let log = event(EventKind::Modify(ModifyKind::Any), "watched/stack.log");
        assert!(!is_relevant_event(&log, exclude));

        let removed = event(EventKind::Remove(RemoveKind::File), "watched/frame_1.png");
        assert!(!is_relevant_event(&removed, exclude));
    }
}
