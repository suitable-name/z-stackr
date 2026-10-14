//! The shortcut table is the single source of truth: it must match the generated
//! block in `docs/shortcuts.md` and the key handler in `ui/app.slint`.

use std::collections::BTreeSet;

use stacker_gui::shortcuts_table::{Context, SHORTCUTS, markdown_table};

const DOC: &str = include_str!("../docs/shortcuts.md");
const APP_SLINT: &str = include_str!("../ui/app.slint");

const BEGIN: &str = "<!-- SHORTCUTS:BEGIN -->\n";
const END: &str = "<!-- SHORTCUTS:END -->";

/// For every table row, the characters `app.slint`'s key handler compares
/// `event.text` with (Escape is matched as `Key.Escape` and has none).
const HANDLED: &[(&str, &[&str])] = &[
    ("Ctrl+N", &["n"]),
    ("Ctrl+Shift+O", &["O"]),
    ("Ctrl+O", &["o"]),
    ("Ctrl+S", &["s"]),
    // Slint reports a shifted letter in upper case.
    ("Ctrl+Shift+S", &["S"]),
    ("Ctrl+E", &["e"]),
    ("Ctrl+,", &[","]),
    ("Ctrl+Z", &["z"]),
    ("Ctrl+Y", &["y", "Z"]),
    // F1 is matched as `Key.F1`, like Esc as `Key.Escape`: no character literal.
    ("F1", &[]),
    ("?", &["?"]),
    ("Esc", &[]),
    ("+", &["+", "="]),
    ("-", &["-"]),
    ("0", &["0"]),
    ("1", &["1"]),
    ("Space", &[" "]),
    // Caps Lock reports the capital, so both are compared.
    ("R", &["r", "R"]),
    ("[", &["["]),
    ("]", &["]"]),
];

/// The text between the generated-block markers, line endings normalised.
fn doc_block() -> String {
    let doc = DOC.replace("\r\n", "\n");
    let start = doc
        .find(BEGIN)
        .expect("docs/shortcuts.md is missing the SHORTCUTS:BEGIN marker")
        + BEGIN.len();
    let length = doc[start..]
        .find(END)
        .expect("docs/shortcuts.md is missing the SHORTCUTS:END marker");
    doc[start..start + length].trim_end_matches('\n').to_owned()
}

/// Every string literal `app.slint` compares `event.text` with.
fn literals_in_app_slint() -> BTreeSet<String> {
    let needle = "event.text == \"";
    let mut found = BTreeSet::new();
    let mut rest = APP_SLINT;
    while let Some(position) = rest.find(needle) {
        rest = &rest[position + needle.len()..];
        let end = rest
            .find('"')
            .expect("unterminated string literal in app.slint");
        found.insert(rest[..end].to_owned());
        rest = &rest[end..];
    }
    found
}

#[test]
fn the_doc_block_matches_the_table() {
    assert_eq!(
        doc_block(),
        markdown_table(),
        "docs/shortcuts.md is out of date: paste markdown_table() between the SHORTCUTS markers"
    );
}

#[test]
fn keys_are_unique_and_rows_are_complete() {
    let mut seen = BTreeSet::new();
    for row in SHORTCUTS {
        assert!(!row.keys.is_empty(), "a shortcut has no keys");
        assert!(!row.action.is_empty(), "{} has no action", row.keys);
        assert!(
            seen.insert(row.keys),
            "{} is bound twice: one key handler serves the whole window",
            row.keys
        );
    }
}

#[test]
fn the_overlay_groups_match_the_known_headings() {
    // `ui/components/shortcut_overlay.slint` renders exactly these two sections.
    for row in SHORTCUTS {
        assert!(
            matches!(row.context.heading(), "Global" | "Image view"),
            "{} is in a section the overlay does not render",
            row.keys
        );
    }
    assert_eq!(Context::Global.heading(), "Global");
    assert_eq!(Context::ImageView.heading(), "Image view");
}

#[test]
fn every_row_has_a_handler_in_app_slint_and_vice_versa() {
    let table_keys: BTreeSet<&str> = SHORTCUTS.iter().map(|row| row.keys).collect();
    let handled_keys: BTreeSet<&str> = HANDLED.iter().map(|(keys, _)| *keys).collect();
    assert_eq!(
        table_keys, handled_keys,
        "the table and the HANDLED list in this test disagree: add or remove the row in both"
    );

    let in_slint = literals_in_app_slint();
    let mut expected = BTreeSet::new();
    for (keys, literals) in HANDLED {
        for literal in *literals {
            assert!(
                in_slint.contains(*literal),
                "{keys} has no `event.text == \"{literal}\"` branch in ui/app.slint"
            );
            expected.insert((*literal).to_owned());
        }
    }
    assert_eq!(
        in_slint, expected,
        "ui/app.slint handles a key that is not in the shortcut table (or the reverse)"
    );
    assert!(
        APP_SLINT.contains("Key.Escape"),
        "ui/app.slint no longer handles Esc"
    );
    assert!(
        APP_SLINT.contains("Key.F1"),
        "ui/app.slint no longer handles F1"
    );
}
