//! The confirmation dialog, driven from Rust with a continuation.
//!
//! A feature asks with [`confirm`] (or [`confirm_with_choice`]) and passes the
//! code that should run if the user agrees. The continuation is parked in a
//! thread-local, keyed by the `confirm-action` id the dialog carries, and the
//! four `DialogModel` callbacks wired in [`setup`] deliver it:
//!
//! * the **primary** (or **secondary**) button runs its continuation once;
//! * **Cancel**, Escape, the close button and a backdrop click drop it unrun;
//! * a callback whose action id no longer matches (a stale answer to a dialog
//!   that was replaced) is ignored.
//!
//! No `RefCell` borrow is held while a continuation runs, so it may start
//! another confirmation, open a picker, or touch the state.

use std::cell::RefCell;

use slint::ComponentHandle;
use stacker_gui::stall_guard::stall_guard;

use super::Ctx;
use crate::{App, DialogModel};

/// What runs when the user agrees: it gets the window and the context.
pub type Handler = Box<dyn FnOnce(&App, &Ctx)>;

/// What the dialog says. `action` identifies the question ("clear-sources"),
/// so a stale answer to a replaced dialog cannot trigger the wrong handler.
#[derive(Debug, Clone, Copy)]
pub struct Prompt<'a> {
    /// The id of the question, passed back by the dialog's callbacks.
    pub action: &'a str,
    /// The dialog title.
    pub heading: &'a str,
    /// The explanation below the title.
    pub message: &'a str,
    /// The label of the confirming button.
    pub primary_label: &'a str,
}

/// How the user answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    /// The confirming button.
    Primary,
    /// The second button of a two-choice question ("Discard").
    Secondary,
    /// The third button; no dialog uses it yet, so it runs nothing.
    Tertiary,
    /// Cancel, Escape, the close button or a backdrop click.
    Cancel,
}

/// The continuations behind the open dialog. Generic over the handler type so
/// the bookkeeping can be tested without a window.
#[derive(Debug)]
pub struct Store<H> {
    parked: Option<Parked<H>>,
}

#[derive(Debug)]
struct Parked<H> {
    action: String,
    primary: Option<H>,
    secondary: Option<H>,
}

impl<H> Store<H> {
    /// An empty store.
    #[must_use]
    pub const fn new() -> Self {
        Self { parked: None }
    }

    /// Park the continuations of the dialog `action`. A dialog that is still
    /// open is replaced: its continuations are dropped unrun.
    pub fn park(&mut self, action: &str, primary: Option<H>, secondary: Option<H>) {
        self.parked = Some(Parked {
            action: action.to_owned(),
            primary,
            secondary,
        });
    }

    /// The answer `choice` to the dialog `action`: the continuation to run, if
    /// any. The dialog is finished with either way, so everything parked for it
    /// is released; an answer to a different (stale) dialog changes nothing.
    pub fn take(&mut self, action: &str, choice: Choice) -> Option<H> {
        if self
            .parked
            .as_ref()
            .is_none_or(|parked| parked.action != action)
        {
            return None;
        }
        let parked = self.parked.take()?;
        match choice {
            Choice::Primary => parked.primary,
            Choice::Secondary => parked.secondary,
            Choice::Tertiary | Choice::Cancel => None,
        }
    }

    /// `true` while a dialog's continuations are parked.
    #[cfg(test)]
    #[must_use]
    pub const fn is_pending(&self) -> bool {
        self.parked.is_some()
    }
}

impl<H> Default for Store<H> {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    /// The continuations behind the dialog on screen.
    static PENDING: RefCell<Store<Handler>> = const { RefCell::new(Store::new()) };
}

/// Wire the dialog's four answers to the parked continuations.
pub fn setup(ui: &App, ctx: &Ctx) {
    // No question is waiting in a new window.
    PENDING.with(|pending| *pending.borrow_mut() = Store::new());
    let dialogs = ui.global::<DialogModel>();
    let answer = |ctx: &Ctx, choice: Choice| {
        let ctx = ctx.clone();
        move |action: slint::SharedString| {
            stall_guard("confirm::answer", || {
                deliver(&ctx, action.as_str(), choice);
            });
        }
    };
    dialogs.on_confirm_primary(answer(ctx, Choice::Primary));
    dialogs.on_confirm_secondary(answer(ctx, Choice::Secondary));
    dialogs.on_confirm_tertiary(answer(ctx, Choice::Tertiary));
    dialogs.on_confirm_cancel(answer(ctx, Choice::Cancel));
}

/// Take the continuation for `choice` out of the store (releasing the borrow)
/// and run it.
fn deliver(ctx: &Ctx, action: &str, choice: Choice) {
    let handler = PENDING.with(|pending| pending.borrow_mut().take(action, choice));
    if let Some(handler) = handler
        && let Some(app) = ctx.ui.upgrade()
    {
        handler(&app, ctx);
    }
}

/// Ask `prompt` with an OK button and Cancel; `on_primary` runs only if the user
/// confirms.
pub fn confirm(ctx: &Ctx, prompt: &Prompt<'_>, on_primary: impl FnOnce(&App, &Ctx) + 'static) {
    open(ctx, prompt, "", Some(Box::new(on_primary)), None);
}

/// Ask `prompt` with a primary, a second (`secondary_label`) and a Cancel button.
///
/// This is the "Save" / "Discard" / Cancel shape of the unsaved-changes guard.
/// Each handler runs only for its own button; Cancel, Escape and the backdrop
/// drop both unrun.
pub fn confirm_with_choice(
    ctx: &Ctx,
    prompt: &Prompt<'_>,
    secondary_label: &str,
    on_primary: impl FnOnce(&App, &Ctx) + 'static,
    on_secondary: impl FnOnce(&App, &Ctx) + 'static,
) {
    open(
        ctx,
        prompt,
        secondary_label,
        Some(Box::new(on_primary)),
        Some(Box::new(on_secondary)),
    );
}

/// Open the dialog for `prompt` with the continuations behind its buttons. A
/// two-button question ("Save" / "Discard") passes a `secondary_label` and a
/// `secondary` handler ([`confirm_with_choice`]); [`confirm`] passes none.
fn open(
    ctx: &Ctx,
    prompt: &Prompt<'_>,
    secondary_label: &str,
    primary: Option<Handler>,
    secondary: Option<Handler>,
) {
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    PENDING.with(|pending| pending.borrow_mut().park(prompt.action, primary, secondary));
    let dialogs = app.global::<DialogModel>();
    dialogs.set_confirm_action(prompt.action.into());
    dialogs.set_confirm_heading(prompt.heading.into());
    dialogs.set_confirm_message(prompt.message.into());
    dialogs.set_confirm_primary_label(prompt.primary_label.into());
    dialogs.set_confirm_secondary_label(secondary_label.into());
    dialogs.set_confirm_tertiary_label("".into());
    dialogs.set_confirm_open(true);
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use super::{Choice, Store};

    /// A handler stand-in that records whether it was dropped.
    struct Probe(Rc<Cell<bool>>);

    impl Drop for Probe {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    fn probe() -> (Probe, Rc<Cell<bool>>) {
        let dropped = Rc::new(Cell::new(false));
        (Probe(Rc::clone(&dropped)), dropped)
    }

    #[test]
    fn a_continuation_is_delivered_exactly_once() {
        let mut store = Store::new();
        store.park("clear", Some(7), None);
        assert!(store.is_pending());
        assert_eq!(store.take("clear", Choice::Primary), Some(7));
        assert_eq!(store.take("clear", Choice::Primary), None);
        assert!(!store.is_pending());
    }

    #[test]
    fn the_secondary_button_delivers_the_secondary_continuation() {
        let mut store = Store::new();
        store.park("save", Some(1), Some(2));
        assert_eq!(store.take("save", Choice::Secondary), Some(2));
        assert!(!store.is_pending(), "answering releases the primary too");
    }

    #[test]
    fn cancel_drops_the_continuations_unrun() {
        let (primary, primary_dropped) = probe();
        let (secondary, secondary_dropped) = probe();
        let mut store = Store::new();
        store.park("save", Some(primary), Some(secondary));
        assert!(store.take("save", Choice::Cancel).is_none());
        assert!(primary_dropped.get());
        assert!(secondary_dropped.get());
        assert!(!store.is_pending());
    }

    #[test]
    fn the_tertiary_button_runs_nothing() {
        let mut store = Store::new();
        store.park("save", Some(1), Some(2));
        assert_eq!(store.take("save", Choice::Tertiary), None);
        assert!(!store.is_pending());
    }

    #[test]
    fn a_stale_answer_changes_nothing() {
        let mut store = Store::new();
        store.park("clear", Some(5), None);
        assert_eq!(store.take("overwrite", Choice::Primary), None);
        assert!(store.is_pending(), "the open dialog keeps its continuation");
        assert_eq!(store.take("clear", Choice::Primary), Some(5));
    }

    #[test]
    fn a_second_dialog_replaces_and_drops_the_first() {
        let (first, first_dropped) = probe();
        let mut store = Store::new();
        store.park("a", Some(first), None);
        let (second, second_dropped) = probe();
        store.park("b", Some(second), None);
        assert!(first_dropped.get());
        assert!(!second_dropped.get());
        assert!(store.take("a", Choice::Primary).is_none());
        assert!(store.take("b", Choice::Primary).is_some());
    }

    #[test]
    fn answering_an_empty_store_is_harmless() {
        let mut store: Store<u8> = Store::new();
        assert_eq!(store.take("anything", Choice::Cancel), None);
    }
}
