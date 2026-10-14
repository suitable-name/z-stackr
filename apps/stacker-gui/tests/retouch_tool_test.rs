//! The Retouch tool's structure in the Slint tree.
//!
//! Retouch used to be a permanent card under the canvas and a left-drag painted
//! whenever a result was shown. It is now a tool: its options exist only while it
//! is on, one function switches it, and the canvas paints only while it is on.
//! These tests read the `.slint` sources, the way the shortcut and layout tests do,
//! so a later edit cannot quietly bring the old arrangement back.

const APP: &str = include_str!("../ui/app.slint");
const CANVAS: &str = include_str!("../ui/components/image_canvas.slint");
const BAR: &str = include_str!("../ui/components/retouch_bar.slint");
const VIEWPORT: &str = include_str!("../ui/models/viewport.slint");

/// How many times `needle` occurs in `haystack`.
fn count(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

#[test]
fn there_is_no_permanent_brush_panel() {
    assert!(
        !APP.contains("BrushSettings"),
        "the always-visible brush card is gone"
    );
    assert!(
        !APP.contains("bottom-height") && !VIEWPORT.contains("bottom-"),
        "no split handle or persisted height for a bottom panel"
    );
}

#[test]
fn the_options_row_is_mounted_only_while_the_tool_is_on() {
    assert_eq!(
        count(APP, "RetouchBar {"),
        1,
        "the row is mounted exactly once"
    );
    assert!(
        APP.contains("if RetouchModel.tool-on: RetouchBar"),
        "the row must sit under `if RetouchModel.tool-on`"
    );
}

#[test]
fn one_function_switches_the_tool() {
    // Every writer (toolbar toggle, `R`, menu, Esc, Done, the result going away)
    // calls `set-retouch-tool`; only that function assigns the property, so
    // `tool-toggled` cannot be skipped.
    assert_eq!(
        count(APP, "RetouchModel.tool-on = "),
        1,
        "only `set-retouch-tool` may assign RetouchModel.tool-on"
    );
    assert_eq!(count(APP, "RetouchModel.tool-toggled(on)"), 1);
    assert!(
        count(APP, "root.set-retouch-tool(") >= 5,
        "toggle, key, menu, Esc, Done and the result-gone case all go through it"
    );
}

#[test]
fn the_tool_needs_a_stacked_result() {
    assert!(
        APP.contains("property <bool> can-retouch: FilesModel.result-files.length > 0;"),
        "the tool is offered only when there is a result"
    );
    assert!(
        APP.contains("changed can-retouch"),
        "losing the result switches the tool off"
    );
}

#[test]
fn the_canvas_paints_only_while_the_tool_is_on() {
    // A left press pans unless the tool is on or Space is held; only the remaining
    // branch starts a stroke.
    assert!(
        CANVAS.contains("(ViewportModel.pan-armed || !root.retouch-on)"),
        "a left press must pan when the tool is off"
    );
    assert_eq!(
        count(CANVAS, "ViewportModel.brush-down("),
        1,
        "one place starts a stroke"
    );
    assert!(
        CANVAS.contains("root.has-images && root.retouch-on && touch.has-hover"),
        "the brush ring is drawn only while the tool is on"
    );
    assert!(
        APP.contains("retouch-on: RetouchModel.tool-on;"),
        "the canvas is told the tool state"
    );
}

#[test]
fn the_row_wraps_instead_of_forcing_a_wide_window() {
    assert!(
        BAR.contains("property <bool> two-rows: ViewportModel.canvas-width <"),
        "the row has a two-line layout for narrow canvases"
    );
    // Choosing the layout from the row's own width is a binding loop (the layout
    // changes the row's horizontal constraints, which decide its width).
    assert!(
        !BAR.contains("two-rows: root.width"),
        "the layout choice must not read the row's own width"
    );
    assert!(
        BAR.contains("done =>") || BAR.contains("callback done()"),
        "the row can be dismissed"
    );
}
