//! The single source of truth for the keyboard shortcuts of the desktop app.
//!
//! One table feeds both the in-app overlay (`ui/components/shortcut_overlay.slint`,
//! filled at start-up by `gui::shortcuts::setup`) and the generated block in
//! `docs/shortcuts.md` ([`markdown_table`]); `tests/shortcuts_test.rs` compares
//! the two so they cannot drift, and checks every row against the key handler
//! in `ui/app.slint`.
//!
//! The key handling itself lives in `app.slint`'s root `FocusScope`: it is
//! the only place that sees key events. All modal layers (settings, export,
//! confirmation, this overlay) make it reject everything but Escape; the `Relief`
//! preview is a non-modal card and leaves the shortcuts alone. The only text
//! fields in the tree sit inside the modal settings and export dialogs, so a
//! typed character can never trigger a shortcut and no separate text-focus
//! gate is needed.
//!
//! The menu bar in `app.slint` shows these keys in its item titles; a `MenuItem`
//! has no accelerator property, so those labels are kept in step by hand.

/// Where a shortcut applies; also the overlay's section heading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Context {
    /// Anywhere in the window.
    Global,
    /// Acting on the image on the canvas.
    ImageView,
}

impl Context {
    /// The section heading the overlay and the generated table use. The overlay
    /// matches on these exact strings.
    #[must_use]
    pub const fn heading(self) -> &'static str {
        match self {
            Self::Global => "Global",
            Self::ImageView => "Image view",
        }
    }
}

/// One keyboard shortcut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shortcut {
    /// The key combination as shown to the user.
    pub keys: &'static str,
    /// What it does.
    pub action: &'static str,
    /// Where it applies.
    pub context: Context,
}

impl Shortcut {
    /// The keys as the overlay shows them. In the browser build Ctrl+N belongs
    /// to the browser (a new window), so New Project answers to Alt+N there;
    /// every other row reads the same everywhere.
    #[must_use]
    pub fn keys_for(&self, web: bool) -> &'static str {
        if web && self.keys == "Ctrl+N" {
            "Alt+N"
        } else {
            self.keys
        }
    }
}

/// Every shortcut `ui/app.slint` implements, in display order.
pub const SHORTCUTS: &[Shortcut] = &[
    Shortcut {
        keys: "Ctrl+N",
        action: "Start a new project",
        context: Context::Global,
    },
    Shortcut {
        keys: "Ctrl+Shift+O",
        action: "Open a project",
        context: Context::Global,
    },
    Shortcut {
        keys: "Ctrl+O",
        action: "Add images",
        context: Context::Global,
    },
    Shortcut {
        keys: "Ctrl+S",
        action: "Save the project",
        context: Context::Global,
    },
    Shortcut {
        keys: "Ctrl+Shift+S",
        action: "Save the project under a new name",
        context: Context::Global,
    },
    Shortcut {
        keys: "Ctrl+E",
        action: "Export the image (opens the export dialog)",
        context: Context::Global,
    },
    Shortcut {
        keys: "Ctrl+,",
        action: "Open the settings",
        context: Context::Global,
    },
    Shortcut {
        keys: "Ctrl+Z",
        action: "Undo the last retouch stroke",
        context: Context::Global,
    },
    Shortcut {
        keys: "Ctrl+Y",
        action: "Redo the last undone stroke (also Ctrl+Shift+Z)",
        context: Context::Global,
    },
    Shortcut {
        keys: "F1",
        action: "Show the manual",
        context: Context::Global,
    },
    Shortcut {
        keys: "?",
        action: "Show this list of shortcuts",
        context: Context::Global,
    },
    Shortcut {
        keys: "Esc",
        action: "Close the open dialog or overlay, leave the Retouch tool, or dismiss the message",
        context: Context::Global,
    },
    Shortcut {
        keys: "+",
        action: "Zoom in (also =)",
        context: Context::ImageView,
    },
    Shortcut {
        keys: "-",
        action: "Zoom out",
        context: Context::ImageView,
    },
    Shortcut {
        keys: "0",
        action: "Fit the image to the window",
        context: Context::ImageView,
    },
    Shortcut {
        keys: "1",
        action: "Show the image at 100 %",
        context: Context::ImageView,
    },
    Shortcut {
        keys: "Space",
        action: "Hold to pan with a left-button drag while the Retouch tool is on (middle button pans too)",
        context: Context::ImageView,
    },
    Shortcut {
        keys: "R",
        action: "Turn the Retouch tool on or off (needs a stacked result)",
        context: Context::ImageView,
    },
    Shortcut {
        keys: "[",
        action: "Smaller brush (Retouch tool on)",
        context: Context::ImageView,
    },
    Shortcut {
        keys: "]",
        action: "Larger brush (Retouch tool on)",
        context: Context::ImageView,
    },
];

/// A Markdown table cell: `|` would end it early.
fn cell(text: &str) -> String {
    text.replace('|', "\\|")
}

/// [`SHORTCUTS`] as one Markdown table (no trailing newline), the exact text
/// between the markers in `docs/shortcuts.md`.
#[must_use]
pub fn markdown_table() -> String {
    let mut out = String::from("| Keys | Action | Where |\n| --- | --- | --- |");
    for row in SHORTCUTS {
        out.push_str("\n| ");
        out.push_str(&cell(row.keys));
        out.push_str(" | ");
        out.push_str(&cell(row.action));
        out.push_str(" | ");
        out.push_str(row.context.heading());
        out.push_str(" |");
    }
    out
}
