use std::process::Command;

#[test]
fn fix_clippy() {
    let status = Command::new("cargo")
        .args([
            "clippy",
            "--fix",
            "--allow-dirty",
            "--allow-no-vcs",
            "--all-targets",
            "--workspace",
        ])
        .status()
        .expect("Failed to execute cargo clippy --fix");

    assert!(status.success());
}
