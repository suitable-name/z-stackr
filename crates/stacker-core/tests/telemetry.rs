use stacker_core::telemetry::*;

/// A valid filter always succeeds: either it installs the subscriber or one is
/// already installed (the initialiser is idempotent).
#[test]
fn test_init_tracing_valid_filter_is_ok() {
    init_tracing("debug").expect("a valid filter must be accepted");
}

/// CORE-7: `init_tracing` is idempotent, so calling it again after the global
/// subscriber is set (by itself, another test, or anyone) returns `Ok`.
#[test]
fn test_init_tracing_twice_is_ok() {
    assert!(init_tracing("info").is_ok());
    assert!(init_tracing("info").is_ok());
    assert!(init_tracing("debug").is_ok());
}

/// CORE-7: an invalid filter argument is reported, not replaced by `info`,
/// and it is reported on every call, including after a subscriber exists.
#[test]
fn test_init_tracing_invalid_filter_is_error() {
    let err = init_tracing("app=notalevel").expect_err("invalid level must be an error");
    assert!(
        format!("{err}").contains("Invalid tracing filter"),
        "unexpected message: {err}"
    );
    // Same answer once a subscriber is registered.
    assert!(init_tracing("info").is_ok());
    assert!(init_tracing("app=notalevel").is_err());
}

/// CORE-7: a rejected filter must not leave a log file or directory behind.
#[test]
fn test_init_tracing_to_file_invalid_filter_creates_nothing() {
    let dir = std::env::temp_dir().join(format!("stacker_telemetry_bad_{}", std::process::id()));
    let log = dir.join("never.log");
    let _ = std::fs::remove_dir_all(&dir);
    let result = init_tracing_to_file(&log, "app=notalevel");
    assert!(result.is_err());
    assert!(
        !dir.exists(),
        "no directory may be created for a bad filter"
    );
}
