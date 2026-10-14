//! The persisted panel layout: round trip, clamping, the `touched` rule, the
//! first-show window fit, and the log path.

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU32, Ordering},
};

use stacker_gui::settings::layout::{
    FILE_NAME, PanelLayout, SIDEBAR_DEFAULT, SIDEBAR_MAX, SIDEBAR_MIN, default_log_path,
    fitted_window_size, layout_path, load, save,
};

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// A fresh, empty directory under the system temp dir.
fn temp_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "z-stackr-layout-test-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

const fn layout(sidebar_width: f32, touched: bool) -> PanelLayout {
    PanelLayout {
        sidebar_width,
        touched,
    }
}

#[test]
fn the_default_is_inside_the_limits_and_untouched() {
    let d = PanelLayout::default();
    assert_eq!(d, d.clamped());
    assert!(!d.touched);
    assert!((SIDEBAR_MIN..=SIDEBAR_MAX).contains(&SIDEBAR_DEFAULT));
}

#[test]
fn the_width_is_clamped_to_its_limits() {
    assert_eq!(layout(10.0, false).clamped(), layout(SIDEBAR_MIN, false));
    assert_eq!(layout(9999.0, true).clamped(), layout(SIDEBAR_MAX, true));
}

#[test]
fn a_non_finite_width_falls_back_to_the_default() {
    let nan = layout(f32::NAN, false).clamped();
    assert!((nan.sidebar_width - SIDEBAR_DEFAULT).abs() < 1e-4);
    // Infinity is not a number either: default, not the upper limit.
    let infinite = layout(f32::INFINITY, false).clamped();
    assert!((infinite.sidebar_width - SIDEBAR_DEFAULT).abs() < 1e-4);
}

#[test]
fn a_user_drag_sets_touched_and_clamps() {
    let after = PanelLayout::default().with_user_width(9999.0);
    assert!(after.touched);
    assert_eq!(after, layout(SIDEBAR_MAX, true));
}

#[test]
fn touched_is_never_cleared_by_later_changes() {
    let once = PanelLayout::default().with_user_width(300.0);
    let twice = once.with_user_width(SIDEBAR_DEFAULT);
    assert!(once.touched && twice.touched);
    // Even dragging back to the default is a choice the user made.
    assert!(twice.touched);
    assert!(!PanelLayout::default().touched);
}

#[test]
fn the_toml_text_round_trips() {
    for original in [
        PanelLayout::default(),
        layout(312.5, true),
        layout(SIDEBAR_MIN, false),
    ] {
        assert_eq!(PanelLayout::parse(&original.to_toml()), original);
    }
}

#[test]
fn a_file_from_a_build_with_a_bottom_panel_still_loads_and_loses_that_key() {
    // Earlier builds also wrote the height of the permanent Retouch panel.
    let old = "sidebar_width = 310.0\nbottom_height = 220.0\ntouched = true\n";
    let loaded = PanelLayout::parse(old);
    assert_eq!(loaded, layout(310.0, true));
    assert!(
        !loaded.to_toml().contains("bottom_height"),
        "the retired key is dropped on the next save"
    );
}

#[test]
fn parsing_never_fails() {
    // Not TOML at all.
    assert_eq!(
        PanelLayout::parse("\u{0}\u{1} not toml ==="),
        PanelLayout::default()
    );
    assert_eq!(PanelLayout::parse(""), PanelLayout::default());
    // A missing key takes its default; the others are kept.
    assert_eq!(
        PanelLayout::parse("sidebar_width = 333.0\n"),
        layout(333.0, false)
    );
    assert_eq!(
        PanelLayout::parse("touched = true\n"),
        layout(SIDEBAR_DEFAULT, true)
    );
    // Integers are accepted, out-of-range values clamped.
    assert_eq!(
        PanelLayout::parse("sidebar_width = 5000\ntouched = true\n"),
        layout(SIDEBAR_MAX, true)
    );
    assert_eq!(
        PanelLayout::parse("sidebar_width = 300\ntouched = true\n"),
        layout(300.0, true)
    );
    // Values of the wrong type are ignored, not fatal.
    assert_eq!(
        PanelLayout::parse("sidebar_width = \"wide\"\ntouched = \"yes\"\n"),
        PanelLayout::default()
    );
}

#[test]
fn a_saved_layout_loads_back() {
    let dir = temp_dir("roundtrip");
    let path = dir.join(FILE_NAME);
    let mine = PanelLayout::default().with_user_width(341.5);
    save(&path, mine).unwrap();
    assert_eq!(load(&path), mine);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn saving_creates_the_directory() {
    let dir = temp_dir("mkdir");
    let path = dir.join("nested").join("deeper").join(FILE_NAME);
    save(&path, PanelLayout::default()).unwrap();
    assert!(path.is_file());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_file_gives_the_defaults() {
    let dir = temp_dir("missing");
    assert_eq!(load(&dir.join(FILE_NAME)), PanelLayout::default());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_corrupt_file_gives_the_defaults_and_the_next_save_repairs_it() {
    let dir = temp_dir("corrupt");
    let path = dir.join(FILE_NAME);
    fs::write(&path, "sidebar_width = [[[ nonsense").unwrap();
    assert_eq!(load(&path), PanelLayout::default());
    let mine = PanelLayout::default().with_user_width(260.0);
    save(&path, mine).unwrap();
    assert_eq!(load(&path), mine);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_layout_file_sits_next_to_the_settings_file() {
    let Some(layout_file) = layout_path() else {
        return; // a machine without a configuration directory
    };
    let settings = stacker_gui::settings::get_config_path().unwrap();
    assert_eq!(layout_file.parent(), settings.parent());
    assert_eq!(layout_file.file_name().unwrap(), FILE_NAME);
}

#[test]
fn the_log_goes_to_an_absolute_per_user_path() {
    let path = default_log_path();
    assert!(path.is_absolute(), "{}", path.display());
    assert_eq!(path.file_name().unwrap(), "stacker.log");
    let cwd = std::env::current_dir().unwrap();
    assert_ne!(path.parent(), Some(cwd.as_path()));
}

#[test]
fn the_window_shrinks_to_the_monitor_and_never_grows() {
    let fractions = (0.94, 0.88);
    // A 1536 x 864 logical monitor (Full HD at 125 %) shrinks a 1680 x 1080 request.
    let fitted = fitted_window_size((1536.0, 864.0), (1680.0, 1080.0), fractions).unwrap();
    assert!((fitted.0 - (1536.0_f32 * 0.94).floor()).abs() < 1e-3);
    assert!((fitted.1 - (864.0_f32 * 0.88).floor()).abs() < 1e-3);
    // A window that already fits is left alone, however big the monitor.
    assert_eq!(
        fitted_window_size((3840.0, 2160.0), (1680.0, 1080.0), fractions),
        None
    );
    assert_eq!(
        fitted_window_size((3840.0, 2160.0), (1400.0, 800.0), fractions),
        None
    );
    // A degenerate monitor changes nothing.
    assert_eq!(
        fitted_window_size((0.0, 0.0), (1680.0, 1080.0), fractions),
        None
    );
    assert_eq!(
        fitted_window_size((f32::NAN, 900.0), (1680.0, 1080.0), fractions),
        None
    );
}

#[test]
fn the_slint_side_starts_from_the_same_numbers() {
    let model = include_str!("../ui/models/viewport.slint");
    for (name, value) in [
        ("sidebar-width", SIDEBAR_DEFAULT),
        ("sidebar-min", SIDEBAR_MIN),
        ("sidebar-max", SIDEBAR_MAX),
    ] {
        let declaration = format!("property <length> {name}: {value}px;");
        assert!(
            model.contains(&declaration),
            "viewport.slint must declare `{declaration}`"
        );
    }
    assert!(
        !model.contains("bottom-"),
        "the bottom panel is gone; viewport.slint must not carry its geometry"
    );
}
