//! The file initialiser installs the process-wide subscriber, so its lifecycle is
//! tested alone in this test binary (one process, one test, fixed order) instead
//! of racing the other telemetry tests for the global slot.

use stacker_core::telemetry::{init_tracing, init_tracing_to_file};

#[test]
fn test_init_tracing_to_file_lifecycle() {
    let dir = std::env::temp_dir().join(format!("stacker_telemetry_file_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let log = dir.join("nested").join("run.log");

    // First call in this process: it must succeed, create the parent directory and
    // the file, and route events into it.
    init_tracing_to_file(&log, "info").expect("no subscriber exists yet, so installing one works");
    assert!(
        log.exists(),
        "log file must exist after init_tracing_to_file"
    );
    tracing::info!("telemetry file test event");
    if std::env::var_os("RUST_LOG").is_none() {
        let text = std::fs::read_to_string(&log).expect("read the log back");
        assert!(
            text.contains("telemetry file test event"),
            "the event must reach the file, got {text:?}"
        );
    }

    // A second file subscriber is refused with the documented message, and the
    // refusal creates nothing.
    let second = dir.join("second").join("run.log");
    let err = init_tracing_to_file(&second, "info")
        .expect_err("a global subscriber is already set, so a second file init fails");
    assert!(
        err.to_string().contains("Failed to initialize tracing"),
        "unexpected error: {err}"
    );
    assert!(!second.parent().is_some_and(std::path::Path::exists));

    // The idempotent stdout initialiser still reports success.
    init_tracing("info").expect("init_tracing is idempotent");

    let _ = std::fs::remove_dir_all(&dir);
}
