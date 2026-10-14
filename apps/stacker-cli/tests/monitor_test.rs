//! Monitor mode against a real file watcher (TST-6): a new frame in the watched
//! directory is a relevant event, a write of the pipeline's own output is not.

use std::{
    path::PathBuf,
    sync::mpsc::{Receiver, channel},
    time::{Duration, Instant},
};

use notify::{Event, RecursiveMode, Watcher};
use stacker_cli::monitor::is_relevant_event;

type Events = Receiver<notify::Result<Event>>;

fn watched_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("stacker_cli_monitor_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::canonicalize(&dir).unwrap()
}

/// Wait up to `timeout` for an event that `keep` accepts and that
/// `is_relevant_event` rates as starting a run.
fn relevant_event_within(
    rx: &Events,
    exclude: &[PathBuf],
    timeout: Duration,
    keep: impl Fn(&Event) -> bool,
) -> bool {
    let end = Instant::now() + timeout;
    while let Some(left) = end.checked_duration_since(Instant::now()) {
        match rx.recv_timeout(left) {
            Ok(Ok(event)) => {
                if keep(&event) && is_relevant_event(&event, exclude) {
                    return true;
                }
            }
            Ok(Err(_)) => {}
            Err(_) => return false,
        }
    }
    false
}

#[test]
fn a_new_frame_starts_a_run_and_the_pipelines_own_output_does_not() {
    let dir = watched_dir();
    let output = dir.join("result.png");
    let exclude = vec![output.clone()];

    let (tx, rx) = channel();
    let mut watcher = notify::recommended_watcher(tx).expect("create the watcher");
    watcher
        .watch(&dir, RecursiveMode::NonRecursive)
        .expect("watch the directory");

    // A frame arrives.
    let frame = dir.join("frame_07.png");
    std::fs::write(&frame, b"not really a png").unwrap();
    let is_frame = |event: &Event| event.paths.iter().any(|p| p.ends_with("frame_07.png"));
    assert!(
        relevant_event_within(&rx, &exclude, Duration::from_secs(10), is_frame),
        "creating a frame must be a relevant event"
    );

    // Let the frame's remaining events settle, then the pipeline writes its
    // result: no event about that file may start another run.
    std::thread::sleep(Duration::from_millis(500));
    while rx.try_recv().is_ok() {}
    std::fs::write(&output, b"result").unwrap();
    let is_output = |event: &Event| event.paths.iter().all(|p| p.ends_with("result.png"));
    assert!(
        !relevant_event_within(&rx, &exclude, Duration::from_millis(1500), is_output),
        "writing the output must not retrigger the pipeline"
    );

    drop(watcher);
    let _ = std::fs::remove_dir_all(&dir);
}
