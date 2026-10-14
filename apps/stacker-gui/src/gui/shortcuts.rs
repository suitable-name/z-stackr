//! Fill the keyboard-shortcut overlay from the table in
//! [`stacker_gui::shortcuts_table`].
//!
//! The key handling itself lives in `ui/app.slint`'s root `FocusScope`; this
//! module only pushes the list the overlay shows, once, at start-up.

use slint::{ComponentHandle, ModelRc, VecModel};
use stacker_gui::shortcuts_table::SHORTCUTS;

use super::Ctx;
use crate::{App, ShortcutItem, ShortcutsModel};

/// Push the shortcut table into the overlay's model (the browser build shows its
/// own key for New Project).
pub fn setup(ui: &App, _ctx: &Ctx) {
    let web = crate::platform::current().is_web();
    let items: Vec<ShortcutItem> = SHORTCUTS
        .iter()
        .map(|row| ShortcutItem {
            keys: row.keys_for(web).into(),
            action: row.action.into(),
            context: row.context.heading().into(),
        })
        .collect();
    ui.global::<ShortcutsModel>()
        .set_shortcuts(ModelRc::new(VecModel::from(items)));
}
