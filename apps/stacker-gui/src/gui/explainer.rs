//! First-use explainers: a one-time card that says what a feature is about, with
//! "Don't show again".
//!
//! A feature calls [`show_once`] at the moment the user first meets it:
//!
//! ```ignore
//! explainer::show_once(ctx, "auto-cull", "Auto-cull is on", "Frames that add almost nothing …");
//! ```
//!
//! The card (`ui/components/explainer.slint`) appears at most once per session
//! for a key, never for a key the user switched off, and never on top of another
//! explainer that is still open (the call then returns `false` and the feature
//! simply asks again the next time). "Don't show again" is kept in `prefs.toml`
//! ([`stacker_gui::settings::prefs`]), written as soon as the card is closed.

use std::{cell::RefCell, collections::BTreeSet};

use slint::ComponentHandle;
use stacker_gui::{
    settings::prefs::{self, Prefs},
    stall_guard::stall_guard,
};

use super::Ctx;
use crate::{App, ExplainerModel};

/// Which explainers may still appear: the persisted "don't show again" set plus
/// the keys already shown in this session. Pure, so it is tested without a window.
#[derive(Debug)]
struct HintBook {
    prefs: Prefs,
    shown_this_session: BTreeSet<String>,
}

impl HintBook {
    const fn new(prefs: Prefs) -> Self {
        Self {
            prefs,
            shown_this_session: BTreeSet::new(),
        }
    }

    /// Claim the right to show `key`: `true` the first time in this session for a
    /// key the user has not switched off. The claim is consumed.
    fn claim(&mut self, key: &str) -> bool {
        if self.prefs.is_suppressed(key) {
            return false;
        }
        self.shown_this_session.insert(key.to_owned())
    }

    /// The card for `key` was closed; `dont_show_again` is its checkbox. Returns
    /// `true` when the persisted set changed (so the file needs writing).
    fn dismissed(&mut self, key: &str, dont_show_again: bool) -> bool {
        dont_show_again && self.prefs.suppress(key)
    }
}

thread_local! {
    /// The hint book of the UI thread; filled by [`setup`].
    static BOOK: RefCell<HintBook> = const {
        RefCell::new(HintBook::new(Prefs {
            suppressed_hints: BTreeSet::new(),
        }))
    };
}

/// Load the hint choices from `prefs.toml` and register the card's dismiss callback.
pub fn setup(ui: &App, _ctx: &Ctx) {
    BOOK.with(|book| *book.borrow_mut() = HintBook::new(prefs::load_stored()));

    ui.global::<ExplainerModel>()
        .on_dismissed(|key, dont_show_again| {
            stall_guard("explainer::dismissed", || {
                let changed =
                    BOOK.with(|book| book.borrow_mut().dismissed(key.as_str(), dont_show_again));
                if changed {
                    persist();
                }
            });
        });
}

/// Write the suppressed set; a failure is logged, never shown (a hint that comes
/// back is no reason to interrupt the user).
fn persist() {
    let snapshot = BOOK.with(|book| book.borrow().prefs.clone());
    if let Err(e) = prefs::save_stored(&snapshot) {
        tracing::warn!("could not save {}: {e}", prefs::FILE_NAME);
    }
}

/// Show the explainer `key` with `heading` and `text`, unless it was shown in
/// this session, the user switched it off, or another explainer is open.
///
/// Returns `true` when the card was put up. UI thread only.
pub fn show_once(ctx: &Ctx, key: &str, heading: &str, text: &str) -> bool {
    let Some(app) = ctx.ui.upgrade() else {
        return false;
    };
    let model = app.global::<ExplainerModel>();
    // One card at a time; the key is not claimed, so it can still appear later.
    if model.get_open() {
        return false;
    }
    if !BOOK.with(|book| book.borrow_mut().claim(key)) {
        return false;
    }
    model.set_key(key.into());
    model.set_heading(heading.into());
    model.set_text(text.into());
    model.set_open(true);
    true
}

#[cfg(test)]
mod tests {
    use stacker_gui::settings::prefs::Prefs;

    use super::HintBook;

    fn book_with(suppressed: &[&str]) -> HintBook {
        let mut prefs = Prefs::default();
        for key in suppressed {
            prefs.suppress(key);
        }
        HintBook::new(prefs)
    }

    #[test]
    fn a_hint_is_claimed_once_per_session() {
        let mut book = book_with(&[]);
        assert!(book.claim("auto-cull"));
        assert!(!book.claim("auto-cull"));
        assert!(book.claim("retouch-sources"), "another key is independent");
    }

    #[test]
    fn a_switched_off_hint_is_never_claimed() {
        let mut book = book_with(&["relief-preview"]);
        assert!(!book.claim("relief-preview"));
        assert!(book.claim("auto-cull"));
    }

    #[test]
    fn dont_show_again_changes_the_persisted_set_once() {
        let mut book = book_with(&[]);
        assert!(
            !book.dismissed("auto-cull", false),
            "unticked: nothing to write"
        );
        assert!(book.dismissed("auto-cull", true));
        assert!(!book.dismissed("auto-cull", true), "already stored");
        assert!(!book.claim("auto-cull"));
    }

    #[test]
    fn closing_without_the_tick_keeps_the_hint_for_the_next_session() {
        let mut book = book_with(&[]);
        assert!(book.claim("auto-cull"));
        assert!(!book.dismissed("auto-cull", false));
        // A new session starts with an empty `shown_this_session`.
        let mut next_session = HintBook::new(book.prefs.clone());
        assert!(next_session.claim("auto-cull"));
    }
}
