//! The export file-name template: every variable, unknown placeholders,
//! sanitising (traversal, reserved Windows names, trailing dots), the numbered
//! suffix that replaces overwriting, and the dialog's live preview.

use std::{
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

use stacker_gui::export::filename_template::{
    DEFAULT_TEMPLATE, FALLBACK_STEM, MAX_STEM_CHARS, TemplateContext, VARIABLES,
    civil_from_unix_seconds, preview, render, resolve_path, sanitize_filename, stem_at,
    unique_path, unknown_placeholders,
};

fn ctx() -> TemplateContext {
    TemplateContext {
        name: "IMG_0042".to_owned(),
        algo: "Apex".to_owned(),
        frames: 37,
        width: 6000,
        height: 4000,
    }
}

/// 2024-03-05 06:07:08 UTC, so the date tests do not depend on the clock.
fn fixed_now() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_709_618_828)
}

fn scratch_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "z-stackr-filename-template-test-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the scratch export folder");
    dir
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .expect("a file name")
        .to_string_lossy()
        .into_owned()
}

// ── Calendar arithmetic ──────────────────────────────────────────────────────

#[test]
fn the_epoch_is_1970_01_01() {
    let civil = civil_from_unix_seconds(0);
    assert_eq!(
        (civil.year, civil.month, civil.day),
        (1970, 1, 1),
        "{civil:?}"
    );
    assert_eq!((civil.hour, civil.minute, civil.second), (0, 0, 0));
}

#[test]
fn a_leap_day_is_recognised() {
    // 2024-02-29 12:00:00 UTC.
    let civil = civil_from_unix_seconds(1_709_208_000);
    assert_eq!(
        (civil.year, civil.month, civil.day, civil.hour),
        (2024, 2, 29, 12)
    );
}

#[test]
fn the_fixed_test_instant_breaks_down_as_expected() {
    let civil = civil_from_unix_seconds(1_709_618_828);
    assert_eq!(
        (
            civil.year,
            civil.month,
            civil.day,
            civil.hour,
            civil.minute,
            civil.second
        ),
        (2024, 3, 5, 6, 7, 8)
    );
}

// ── render ───────────────────────────────────────────────────────────────────

#[test]
fn every_listed_variable_expands() {
    for name in VARIABLES {
        let rendered = render(&format!("{{{name}}}"), &ctx(), fixed_now());
        assert!(
            !rendered.contains('{') && !rendered.contains('}'),
            "{{{name}}} stayed literal: {rendered:?}"
        );
    }
}

#[test]
fn text_variables_expand_verbatim() {
    assert_eq!(
        render("{name}-{algo}", &ctx(), fixed_now()),
        "IMG_0042-Apex"
    );
}

#[test]
fn numeric_variables_expand_as_decimal() {
    assert_eq!(render("{n}f_{w}x{h}", &ctx(), fixed_now()), "37f_6000x4000");
}

#[test]
fn date_time_and_timestamp_use_dashes_and_utc() {
    assert_eq!(
        render("{date}_{time}_{timestamp}", &ctx(), fixed_now()),
        "2024-03-05_06-07-08_1709618828"
    );
}

#[test]
fn an_unknown_placeholder_stays_literal() {
    assert_eq!(
        render("{name}_{nope}", &ctx(), fixed_now()),
        "IMG_0042_{nope}"
    );
}

#[test]
fn an_unterminated_brace_stays_literal_and_does_not_panic() {
    assert_eq!(
        render("{name}_{oops", &ctx(), fixed_now()),
        "IMG_0042_{oops"
    );
}

#[test]
fn empty_braces_stay_literal() {
    assert_eq!(render("a{}b", &ctx(), fixed_now()), "a{}b");
}

#[test]
fn a_template_without_variables_is_unchanged() {
    assert_eq!(render("plain-text", &ctx(), fixed_now()), "plain-text");
}

// ── unknown placeholders ─────────────────────────────────────────────────────

#[test]
fn known_variables_are_not_reported() {
    assert_eq!(
        unknown_placeholders("{name}_{algo}_{date}_{time}_{timestamp}_{n}_{w}x{h}"),
        Vec::<String>::new()
    );
}

#[test]
fn unknown_placeholders_are_reported_once_in_order() {
    assert_eq!(
        unknown_placeholders("{foo}{name}{bar}{foo}"),
        vec!["{foo}".to_owned(), "{bar}".to_owned()]
    );
}

#[test]
fn an_unterminated_brace_is_reported_as_typed() {
    assert_eq!(unknown_placeholders("{name}_{da"), vec!["{da".to_owned()]);
}

// ── sanitize_filename ────────────────────────────────────────────────────────

#[test]
fn forbidden_characters_become_underscores() {
    let cleaned = sanitize_filename("a<b>c:d\"e/f\\g|h?i*j");
    assert_eq!(cleaned, "a_b_c_d_e_f_g_h_i_j");
}

#[test]
fn control_characters_are_replaced() {
    assert_eq!(sanitize_filename("a\u{0}b\tc"), "a_b_c");
}

#[test]
fn traversal_cannot_leave_a_single_path_component() {
    for attack in [
        "../../etc/passwd",
        "..\\..\\windows\\system32",
        "/abs/olute",
        "..",
    ] {
        let cleaned = sanitize_filename(attack);
        assert!(
            !cleaned.contains('/') && !cleaned.contains('\\'),
            "{attack:?} -> {cleaned:?}"
        );
        assert_ne!(cleaned, "..", "{attack:?}");
        assert_ne!(cleaned, ".", "{attack:?}");
        assert!(!cleaned.is_empty(), "{attack:?}");
    }
}

#[test]
fn reserved_windows_device_names_are_prefixed() {
    for reserved in ["CON", "con", "Nul", "COM1", "lpt9", "AUX.txt", "PRN"] {
        let cleaned = sanitize_filename(reserved);
        assert!(
            cleaned.starts_with('_'),
            "{reserved:?} was kept as {cleaned:?}"
        );
    }
    // Only the whole stem is reserved: a longer name is fine.
    assert_eq!(sanitize_filename("CONSOLE"), "CONSOLE");
    assert_eq!(sanitize_filename("COM10"), "COM10");
}

#[test]
fn trailing_dots_and_spaces_are_trimmed_but_inner_ones_stay() {
    assert_eq!(sanitize_filename("R. Strauss. "), "R. Strauss");
    assert_eq!(sanitize_filename("  lead"), "lead");
}

#[test]
fn nothing_left_falls_back_to_a_valid_name() {
    for empty in ["", "...", " . ", "   "] {
        assert_eq!(sanitize_filename(empty), FALLBACK_STEM, "{empty:?}");
    }
}

#[test]
fn a_very_long_name_is_cut_without_a_trailing_dot() {
    let long = format!("{}.{}", "a".repeat(MAX_STEM_CHARS - 1), "b".repeat(50));
    let cleaned = sanitize_filename(&long);
    assert!(
        cleaned.chars().count() <= MAX_STEM_CHARS,
        "{}",
        cleaned.len()
    );
    assert!(!cleaned.ends_with('.'), "{cleaned:?}");
}

#[test]
fn accented_and_unicode_names_survive() {
    assert_eq!(sanitize_filename("Käfer_蜂"), "Käfer_蜂");
}

// ── The finished stem and the preview ────────────────────────────────────────

#[test]
fn the_default_template_names_the_stem_after_the_source() {
    assert_eq!(
        stem_at(DEFAULT_TEMPLATE, &ctx(), fixed_now()),
        "IMG_0042_stacked"
    );
}

#[test]
fn a_blank_template_uses_the_default() {
    assert_eq!(stem_at("   ", &ctx(), fixed_now()), "IMG_0042_stacked");
    assert_eq!(stem_at("", &ctx(), fixed_now()), "IMG_0042_stacked");
}

#[test]
fn a_hostile_variable_value_is_sanitised_after_expansion() {
    let mut hostile = ctx();
    hostile.name = "../../CON".to_owned();
    let stem = stem_at("{name}", &hostile, fixed_now());
    assert!(!stem.contains('/') && !stem.contains('\\'), "{stem:?}");
    assert_ne!(stem.to_ascii_uppercase(), "CON");
}

#[test]
fn the_preview_matches_the_name_an_export_would_get() {
    // Both use the clock, so only the clock-free part of the template is compared.
    assert_eq!(preview("{name}_{algo}_{n}", &ctx()), "IMG_0042_Apex_37");
}

// ── unique_path and resolve_path ─────────────────────────────────────────────

#[test]
fn a_free_name_is_used_as_it_is() {
    let dir = scratch_dir("free");
    let path = unique_path(&dir, "result", "png");
    assert_eq!(file_name(&path), "result.png");
    assert_eq!(path.parent(), Some(dir.as_path()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_existing_file_is_never_overwritten() {
    let dir = scratch_dir("collisions");
    let first = unique_path(&dir, "result", "tiff");
    std::fs::write(&first, b"one").expect("write the first file");

    let second = unique_path(&dir, "result", "tiff");
    assert_eq!(file_name(&second), "result (2).tiff");
    std::fs::write(&second, b"two").expect("write the second file");

    let third = unique_path(&dir, "result", "tiff");
    assert_eq!(file_name(&third), "result (3).tiff");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_empty_extension_gives_a_name_without_a_dot() {
    let dir = scratch_dir("no-ext");
    assert_eq!(file_name(&unique_path(&dir, "result", "")), "result");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn resolve_path_stays_directly_inside_the_folder() {
    let dir = scratch_dir("traversal");
    let mut hostile = ctx();
    hostile.name = "../../../etc/passwd".to_owned();
    let path = resolve_path(&dir, "{name}", &hostile, "png");
    assert_eq!(path.parent(), Some(dir.as_path()), "{path:?}");
    assert_eq!(path.extension().and_then(|ext| ext.to_str()), Some("png"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn resolve_path_numbers_a_repeated_export() {
    let dir = scratch_dir("repeat");
    let first = resolve_path(&dir, "{name}_{algo}", &ctx(), "jpg");
    assert_eq!(file_name(&first), "IMG_0042_Apex.jpg");
    std::fs::write(&first, b"x").expect("write the first export");
    let second = resolve_path(&dir, "{name}_{algo}", &ctx(), "jpg");
    assert_eq!(file_name(&second), "IMG_0042_Apex (2).jpg");
    let _ = std::fs::remove_dir_all(&dir);
}
