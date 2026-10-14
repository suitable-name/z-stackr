//! Help: the bundled manual, the About card and File > Quit.
//!
//! The manual is `docs/manual.md`, compiled into the binary with `include_str!`
//! and parsed at start-up into [`ManualBlock`]s that
//! `ui/components/manual_dialog.slint` renders. There is no shell call and no
//! external viewer: Help > Manual works offline and from a binary copied alone.
//!
//! Only the small Markdown subset the manual uses is understood (see
//! [`parse_manual`]): `#`/`##`/`###` headings, `-` bullets, fenced code blocks,
//! paragraphs, and inline `**bold**`, `` `code` `` and `[text](url)` (shown as
//! plain text).

use slint::{ComponentHandle, ModelRc, VecModel, platform::WindowEvent};
use stacker_gui::stall_guard::stall_guard;

use super::Ctx;
use crate::{App, ManualBlock, ManualModel};

/// The manual's source text.
const MANUAL: &str = include_str!("../../docs/manual.md");

/// Fill the manual and About card, and register File > Quit.
pub fn setup(ui: &App, _ctx: &Ctx) {
    let model = ui.global::<ManualModel>();
    let blocks: Vec<ManualBlock> = parse_manual(MANUAL)
        .into_iter()
        .map(|block| ManualBlock {
            kind: block.kind.name().into(),
            text: block.text.into(),
        })
        .collect();
    model.set_blocks(ModelRc::new(VecModel::from(blocks)));
    model.set_version(env!("CARGO_PKG_VERSION").into());
    model.set_licence(env!("CARGO_PKG_LICENSE").into());

    // Quit asks the window to close, exactly like the close button, so a guard on
    // the close request (unsaved changes) sees it too.
    let weak = ui.as_weak();
    ui.on_quit_app(move || {
        stall_guard("help::quit-app", || {
            if let Some(app) = weak.upgrade() {
                app.window().dispatch_event(WindowEvent::CloseRequested);
            }
        });
    });
}

/// What a block of the manual is; the name is what the Slint side switches on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Heading1,
    Heading2,
    Heading3,
    Paragraph,
    Bullet,
    Code,
}

impl BlockKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Heading1 => "h1",
            Self::Heading2 => "h2",
            Self::Heading3 => "h3",
            Self::Paragraph => "p",
            Self::Bullet => "li",
            Self::Code => "code",
        }
    }
}

/// One rendered block.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Block {
    kind: BlockKind,
    text: String,
}

impl Block {
    fn new(kind: BlockKind, text: &str) -> Self {
        Self {
            kind,
            text: inline_text(text),
        }
    }
}

/// `text` without the inline Markdown marks: `**` and `` ` `` are dropped and
/// `[label](target)` becomes `label`.
fn inline_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if let Some(after) = rest.strip_prefix("**") {
            rest = after;
        } else if c == '`' {
            rest = &rest[1..];
        } else if c == '['
            && let Some((label, after)) = split_link(rest)
        {
            out.push_str(label);
            rest = after;
        } else {
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

/// `[label](target)` at the start of `text`: the label and what follows it.
fn split_link(text: &str) -> Option<(&str, &str)> {
    let after_open = text.strip_prefix('[')?;
    let (label, after_label) = after_open.split_once("](")?;
    let (_target, after) = after_label.split_once(')')?;
    (!label.contains('[')).then_some((label, after))
}

/// Parse the manual's Markdown subset into blocks.
///
/// * `# `, `## `, `### ` start a heading;
/// * `- ` or `* ` starts a bullet (one block per line);
/// * a fenced block (a line of three backticks opens and closes it) gives one
///   code block per line;
/// * consecutive other lines form one paragraph, ended by a blank line;
/// * HTML comment lines (`<!-- ... -->`) are dropped.
fn parse_manual(text: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut paragraph: Vec<&str> = Vec::new();
    let mut in_code = false;

    let flush = |paragraph: &mut Vec<&str>, blocks: &mut Vec<Block>| {
        if !paragraph.is_empty() {
            blocks.push(Block::new(BlockKind::Paragraph, &paragraph.join(" ")));
            paragraph.clear();
        }
    };

    for raw in text.lines() {
        let line = raw.trim_end();
        if line.trim_start().starts_with("```") {
            flush(&mut paragraph, &mut blocks);
            in_code = !in_code;
            continue;
        }
        if in_code {
            // Code keeps its text as written: no inline marks are stripped.
            blocks.push(Block {
                kind: BlockKind::Code,
                text: line.to_owned(),
            });
            continue;
        }
        let trimmed = line.trim_start();
        if trimmed.is_empty() || (trimmed.starts_with("<!--") && trimmed.ends_with("-->")) {
            flush(&mut paragraph, &mut blocks);
        } else if let Some(heading) = trimmed.strip_prefix("### ") {
            flush(&mut paragraph, &mut blocks);
            blocks.push(Block::new(BlockKind::Heading3, heading));
        } else if let Some(heading) = trimmed.strip_prefix("## ") {
            flush(&mut paragraph, &mut blocks);
            blocks.push(Block::new(BlockKind::Heading2, heading));
        } else if let Some(heading) = trimmed.strip_prefix("# ") {
            flush(&mut paragraph, &mut blocks);
            blocks.push(Block::new(BlockKind::Heading1, heading));
        } else if let Some(item) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            flush(&mut paragraph, &mut blocks);
            blocks.push(Block::new(BlockKind::Bullet, item));
        } else {
            paragraph.push(trimmed);
        }
    }
    flush(&mut paragraph, &mut blocks);
    blocks
}

#[cfg(test)]
mod tests {
    use super::{BlockKind, MANUAL, inline_text, parse_manual};

    fn kinds(text: &str) -> Vec<BlockKind> {
        parse_manual(text).iter().map(|b| b.kind).collect()
    }

    #[test]
    fn headings_paragraphs_and_bullets_are_told_apart() {
        let text = "# Title\n\nSome words\nthat wrap.\n\n## Part\n\n- one\n- two\n\n### Sub\nx\n";
        assert_eq!(
            kinds(text),
            vec![
                BlockKind::Heading1,
                BlockKind::Paragraph,
                BlockKind::Heading2,
                BlockKind::Bullet,
                BlockKind::Bullet,
                BlockKind::Heading3,
                BlockKind::Paragraph,
            ]
        );
    }

    #[test]
    fn the_lines_of_a_paragraph_are_joined_with_spaces() {
        let blocks = parse_manual("first line\nsecond line\n");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "first line second line");
    }

    #[test]
    fn a_fenced_block_gives_one_code_block_per_line_untouched() {
        let blocks = parse_manual("```\nkey = \"**value**\"\n\nnext\n```\nafter\n");
        let code: Vec<&str> = blocks
            .iter()
            .filter(|b| b.kind == BlockKind::Code)
            .map(|b| b.text.as_str())
            .collect();
        assert_eq!(code, vec!["key = \"**value**\"", "", "next"]);
        assert_eq!(blocks.last().map(|b| b.kind), Some(BlockKind::Paragraph));
    }

    #[test]
    fn html_comments_are_dropped() {
        let blocks = parse_manual("<!-- hidden -->\ntext\n");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "text");
    }

    #[test]
    fn inline_marks_are_removed() {
        assert_eq!(
            inline_text("Press **F1** or `Ctrl+E`, see [the docs](https://x.y/z) now."),
            "Press F1 or Ctrl+E, see the docs now."
        );
    }

    #[test]
    fn a_bracket_that_is_not_a_link_stays() {
        assert_eq!(inline_text("a [b] c"), "a [b] c");
        assert_eq!(inline_text("a [b]( c"), "a [b]( c");
    }

    #[test]
    fn the_bundled_manual_has_a_title_and_the_main_sections() {
        let blocks = parse_manual(MANUAL);
        assert_eq!(blocks.first().map(|b| b.kind), Some(BlockKind::Heading1));
        let headings: Vec<&str> = blocks
            .iter()
            .filter(|b| b.kind == BlockKind::Heading2)
            .map(|b| b.text.as_str())
            .collect();
        for wanted in [
            "Workflow",
            "Alignment modes",
            "Fusion methods",
            "Auto-cull, Sort and Cull",
            "Retouch",
            "Export",
            "Settings file",
            "Keyboard shortcuts",
        ] {
            assert!(
                headings.contains(&wanted),
                "the manual has no \"{wanted}\" section: {headings:?}"
            );
        }
    }

    #[test]
    fn the_bundled_manual_leaves_no_markup_behind() {
        for block in parse_manual(MANUAL) {
            if block.kind == BlockKind::Code {
                continue;
            }
            assert!(!block.text.contains("**"), "{}", block.text);
            assert!(!block.text.contains('`'), "{}", block.text);
        }
    }
}
