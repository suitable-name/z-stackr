use std::process::Command;

#[test]
fn fix_fmt() {
    let status = Command::new("cargo")
        .args(["fmt"])
        .status()
        .expect("Failed to execute cargo fmt");

    assert!(status.success());
}
