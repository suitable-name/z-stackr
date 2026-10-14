//! The scenarios, in the order they run.

use super::{
    ScenarioFn,
    flows::*,
    harness::{Outcome, Ui, check},
};

macro_rules! scenarios {
    ($($f:ident),* $(,)?) => {
        pub const ALL: &[ScenarioFn] = &[
            $((stringify!($f), || Box::pin($f())),)*
        ];
    };
}

scenarios![
    open_and_stack,
    retouch_tool,
    retouch_blocked_off_the_newest_result,
    retouch_tool_ends_with_the_result,
    keys_survive_the_retouch_row,
    esc_leaves_the_tool_before_a_toast,
    project_round_trip,
    unsaved_changes_guard,
    quit_from_the_menu,
    export_flow,
    settings_dialog,
    relief_preview,
    cancel_from_the_chip,
    edit_sources_during_a_run,
    source_list_editing,
    dropped_files,
    zoom_controls,
    first_use_hints,
    align_and_stale,
    stale_result,
    live_monitor,
    disabled_menu_items_do_nothing,
    recent_projects,
    algorithms,
    strata_run,
    all_three_run,
    sort_and_cull,
    settings_tabs,
    action_matrix,
];

/// The smallest end-to-end path: add frames, stack, the tool becomes available.
async fn open_and_stack() -> Outcome {
    let ui = Ui::boot("open-and-stack");
    check!(ui.files() == 0, "a fresh app has no files");
    check!(
        !ui.enabled("Start Stacking")?,
        "Start Stacking is off without files"
    );
    check!(!ui.enabled("Retouch")?, "Retouch is off without a result");
    ui.load_frames(3).await?;
    check!(
        ui.enabled("Start Stacking")?,
        "Start Stacking is on with files"
    );
    ui.stack().await?;
    check!(ui.results() == 1, "one result");
    check!(
        ui.enabled("Retouch")?,
        "Retouch is offered once there is a result"
    );
    Ok(())
}

/// Every action in every state (see `matrix.rs`).
async fn action_matrix() -> Outcome {
    super::matrix::run().await
}
