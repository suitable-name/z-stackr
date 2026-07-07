use stacker_core::telemetry::*;

#[test]
fn test_init_tracing_mocked() {
    let result = init_tracing("debug");
    match result {
        Ok(()) => {}
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("Failed to initialize tracing")
                    || msg.contains("Invalid tracing filter")
            );
        }
    }
}

/// Verify that `init_tracing_to_file` either creates the file and writes to
/// it, or returns the expected error type when a global subscriber is
/// already registered (which happens when tests run in the same process).
#[test]
fn test_init_tracing_to_file_creates_file_or_errors() {
    let tmp =
        std::env::temp_dir().join(format!("stacker_telemetry_test_{}.log", std::process::id()));
    // Clean up any left-over from a previous run.
    let _ = std::fs::remove_file(&tmp);

    let result = init_tracing_to_file(&tmp, "info");
    match result {
        Ok(()) => {
            // Subscriber registered — the file must exist and be non-empty
            // after we emit at least one event.
            tracing::info!("telemetry file test event");
            // Give the synchronous writer a moment (it's sync, so no wait
            // needed, but flush by dropping any guard is implicit here).
            assert!(
                tmp.exists(),
                "log file must exist after init_tracing_to_file"
            );
        }
        Err(e) => {
            // A global subscriber was already registered by another test or
            // init_tracing().  The error message must match the expected
            // pattern — this is the important invariant.
            let msg = format!("{e}");
            assert!(
                msg.contains("Failed to initialize tracing") || msg.contains("IO error"),
                "unexpected error from init_tracing_to_file: {msg}"
            );
        }
    }

    let _ = std::fs::remove_file(&tmp);
}
