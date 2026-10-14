use std::{
    cell::{Cell, RefCell},
    path::PathBuf,
    rc::Rc,
    sync::{Arc, Mutex, mpsc},
    thread,
    time::{Duration, Instant},
};

use stacker_gui::{
    jobs_core::{
        ActivityList, CancelFlag, CancelMap, INDETERMINATE, JobError, RunQueue, StagePlan,
        fraction_of, monitor_label, run_guarded, run_job,
    },
    sync::lock,
};

/// Long enough for a loaded machine, short enough that a hang fails the test.
const WAIT: Duration = Duration::from_secs(10);

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

// ── Activity list: order and ids ────────────────────────────────────────────

#[test]
fn ids_increase_and_entries_stay_in_start_order() {
    let mut list = ActivityList::default();
    let a = list.start("align", "Align", true, true, 0);
    let b = list.start("stack", "Stack", true, true, 10);
    let c = list.start("monitor", "Monitor", true, false, 20);
    assert_eq!((a, b, c), (1, 2, 3));
    let kinds: Vec<&str> = list.entries().iter().map(|e| e.kind.as_str()).collect();
    assert_eq!(kinds, ["align", "stack", "monitor"]);
}

#[test]
fn finishing_the_middle_entry_keeps_the_order_of_the_rest() {
    let mut list = ActivityList::default();
    let a = list.start("a", "A", false, true, 0);
    let b = list.start("b", "B", false, true, 0);
    let c = list.start("c", "C", false, true, 0);
    assert!(list.finish(b));
    let ids: Vec<u64> = list.entries().iter().map(|e| e.id).collect();
    assert_eq!(ids, [a, c]);
    assert!(!list.finish(b), "a second finish is a harmless no-op");
    assert!(!list.finish(999), "an unknown id is a harmless no-op");
}

#[test]
fn ids_are_never_reused() {
    let mut list = ActivityList::default();
    let a = list.start("a", "A", false, true, 0);
    list.finish(a);
    let b = list.start("b", "B", false, true, 0);
    assert_ne!(a, b);
}

#[test]
fn a_new_activity_is_indeterminate_and_progress_updates_only_its_entry() {
    let mut list = ActivityList::default();
    let a = list.start("a", "A", false, true, 0);
    let b = list.start("b", "B", false, true, 0);
    assert!(near(list.entries()[0].progress, INDETERMINATE));
    assert!(list.set_progress(b, 0.5, Instant::now()));
    assert!(near(list.entries()[0].progress, INDETERMINATE));
    assert!(near(list.entries()[1].progress, 0.5));
    assert!(!list.set_progress(a + b + 100, 0.5, Instant::now()));
}

#[test]
fn only_blocking_activities_count_as_busy() {
    let mut list = ActivityList::default();
    assert_eq!(list.blocking_count(), 0);
    let monitor = list.start("monitor", "Monitor: watching", true, false, 0);
    assert_eq!(
        list.blocking_count(),
        0,
        "the watcher must not disable Stop"
    );
    assert_eq!(list.first_blocking_label(), None);
    let run = list.start("stack", "Stack", true, true, 0);
    assert_eq!(list.blocking_count(), 1);
    assert_eq!(list.first_blocking_label(), Some("Stack"));
    list.finish(run);
    list.finish(monitor);
    assert!(list.is_empty());
}

#[test]
fn a_cancel_request_turns_the_chip_into_cancelling() {
    let mut list = ActivityList::default();
    let id = list.start("stack", "Stack", true, true, 0);
    assert!(list.mark_cancelling(id));
    let entry = &list.entries()[0];
    assert!(!entry.cancellable, "a second click has nothing to press");
    assert_eq!(entry.label, "Stack (cancelling…)");
    assert!(!list.mark_cancelling(999));
}

#[test]
fn set_label_replaces_the_chip_text() {
    let mut list = ActivityList::default();
    let id = list.start("monitor", "Monitor: watching", true, false, 0);
    assert!(list.set_label(id, &monitor_label(2, true)));
    assert_eq!(list.entries()[0].label, "Monitor: 2 queued");
}

#[test]
fn the_eta_text_follows_the_reported_progress() {
    let start = Instant::now();
    let mut list = ActivityList::default();
    let id = list.start("stack", "Stack", true, true, 0);
    assert_eq!(list.eta_text(id, start), "", "no estimate without history");
    for t in 0..=10_u32 {
        list.set_progress(
            id,
            0.05 * f32::from(u16::try_from(t).unwrap_or(0)),
            start + Duration::from_secs(u64::from(t)),
        );
    }
    // 0.5 at t = 10 s with 0.05/s: 10 s left.
    assert_eq!(
        list.eta_text(id, start + Duration::from_secs(10)),
        "about 10 s left"
    );
    assert_eq!(list.eta_text(id + 1, start), "", "unknown id");
}

// ── Cancel map ──────────────────────────────────────────────────────────────

#[test]
fn invoking_a_registered_cancel_runs_the_closure_once() {
    let calls = Rc::new(Cell::new(0));
    let mut map = CancelMap::default();
    map.insert(1, {
        let calls = Rc::clone(&calls);
        Rc::new(move || calls.set(calls.get() + 1))
    });
    assert!(map.invoke(1));
    assert_eq!(calls.get(), 1);
}

#[test]
fn invoking_an_unknown_or_removed_id_does_nothing() {
    let calls = Rc::new(Cell::new(0));
    let mut map = CancelMap::default();
    assert!(!map.invoke(42));
    map.insert(1, {
        let calls = Rc::clone(&calls);
        Rc::new(move || calls.set(calls.get() + 1))
    });
    map.remove(1);
    assert!(!map.invoke(1));
    assert_eq!(calls.get(), 0, "a removed cancel must never fire");
}

#[test]
fn a_cancel_closure_can_finish_its_own_activity() {
    // The registry hands out a clone before running the closure, so the closure
    // may remove itself from the map (the monitor's cancel does exactly that).
    let map = Rc::new(RefCell::new(CancelMap::default()));
    let ran = Rc::new(Cell::new(false));
    {
        let inner = Rc::clone(&map);
        let ran = Rc::clone(&ran);
        map.borrow_mut().insert(
            7,
            Rc::new(move || {
                inner.borrow_mut().remove(7);
                ran.set(true);
            }),
        );
    }
    let cancel = map.borrow().get(7).expect("registered");
    cancel();
    assert!(ran.get());
    assert!(map.borrow().get(7).is_none());
}

// ── Cancel flag and the job body ────────────────────────────────────────────

#[test]
fn the_cancel_flag_is_shared_between_clones() {
    let flag = CancelFlag::default();
    let worker_side = flag.clone();
    assert!(worker_side.check("anywhere").is_ok());
    flag.cancel();
    flag.cancel();
    assert!(worker_side.is_cancelled());
    assert_eq!(
        worker_side.check("after loading"),
        Err(JobError::Cancelled("after loading".to_owned()))
    );
}

/// A panicking body still finishes its activity and reports: `finish` runs on
/// every exit path.
#[test]
fn a_panicking_job_still_finishes_its_activity_and_reports() {
    let list = Arc::new(Mutex::new(ActivityList::default()));
    let id = lock(&list).start("stack", "Stack", true, true, 0);
    let reported: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let cancel = CancelFlag::default();
    let worker = {
        let list = Arc::clone(&list);
        let reported = Arc::clone(&reported);
        thread::spawn(move || {
            run_job(
                &cancel,
                |_| -> Result<(), JobError> { panic!("boom in the fusion") },
                |result| {
                    lock(&list).finish(id);
                    let message = result.map_or_else(|e| e.message("Stack"), |()| String::new());
                    *lock(&reported) = Some(message);
                },
            );
        })
    };
    worker.join().expect("the panic was caught inside the job");

    assert!(lock(&list).is_empty(), "the activity must be finished");
    let message = lock(&reported).clone().expect("the job reported");
    assert!(message.contains("boom in the fusion"), "{message}");
    assert!(
        message.starts_with("Stack stopped unexpectedly"),
        "{message}"
    );
}

#[test]
fn a_panic_with_a_formatted_message_keeps_its_text() {
    let value = 7;
    let result: Result<(), JobError> = run_guarded(|| panic!("frame {value} is broken"));
    assert_eq!(
        result,
        Err(JobError::Panicked("frame 7 is broken".to_owned()))
    );
}

#[test]
fn a_successful_job_hands_its_value_to_finish() {
    let (tx, rx) = mpsc::channel();
    run_job(
        &CancelFlag::default(),
        |_| Ok(21 * 2),
        |result| tx.send(result).expect("receiver alive"),
    );
    assert_eq!(rx.recv_timeout(WAIT), Ok(Ok(42)));
}

/// The fake two-stage job: stage one finishes, the cancel arrives, the flag is
/// seen between the stages, and stage two never runs.
#[test]
fn a_cancel_between_stages_stops_before_the_second_stage() {
    let cancel = CancelFlag::default();
    let stage_two_ran = Arc::new(Mutex::new(false));
    let plan = StagePlan::new(&[("one", 1.0), ("two", 1.0)]);
    let reached = Arc::new(Mutex::new(Vec::<f32>::new()));

    let (done_tx, done_rx) = mpsc::channel();
    let (stage_one_done_tx, stage_one_done_rx) = mpsc::channel::<()>();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let worker_cancel = cancel.clone();
    let worker = {
        let stage_two_ran = Arc::clone(&stage_two_ran);
        let reached = Arc::clone(&reached);
        thread::spawn(move || {
            run_job(
                &worker_cancel,
                |cancel| {
                    lock(&reached).push(plan.fraction(0, 1.0));
                    stage_one_done_tx.send(()).expect("test alive");
                    go_rx.recv().expect("test alive");
                    cancel.check("between the stages")?;
                    *lock(&stage_two_ran) = true;
                    Ok(())
                },
                |result| done_tx.send(result).expect("test alive"),
            );
        })
    };

    stage_one_done_rx.recv_timeout(WAIT).expect("stage one ran");
    cancel.cancel();
    go_tx.send(()).expect("worker alive");
    let result = done_rx.recv_timeout(WAIT).expect("the job finished");
    worker.join().expect("no panic");

    assert_eq!(
        result,
        Err(JobError::Cancelled("between the stages".to_owned()))
    );
    assert!(
        !*lock(&stage_two_ran),
        "stage two must not run after a cancel"
    );
    assert!(near(lock(&reached)[0], 0.5));
}

#[test]
fn a_cancel_inside_a_long_stage_is_seen_at_the_next_frame() {
    // A stage that checks the flag once per frame stops at the next frame.
    fn align_ten_frames(cancel: &CancelFlag, processed: &mut u32) -> Result<(), JobError> {
        for frame in 0..10 {
            cancel.check("while aligning")?;
            *processed += 1;
            if frame == 3 {
                cancel.cancel();
            }
        }
        Ok(())
    }
    let cancel = CancelFlag::default();
    let mut processed = 0;
    let result = align_ten_frames(&cancel, &mut processed);
    assert_eq!(processed, 4);
    assert_eq!(
        result,
        Err(JobError::Cancelled("while aligning".to_owned()))
    );
}

// ── Messages ────────────────────────────────────────────────────────────────

#[test]
fn error_messages_read_like_sentences() {
    assert_eq!(
        JobError::Cancelled("while aligning".to_owned()).message("Stack"),
        "Stack cancelled while aligning."
    );
    assert_eq!(
        JobError::Cancelled(String::new()).message("Align"),
        "Align cancelled."
    );
    assert_eq!(
        JobError::Failed("Align: no images.".to_owned()).message("Align"),
        "Align: no images."
    );
    assert!(
        JobError::Panicked("kaboom".to_owned())
            .message("Stack")
            .contains("kaboom")
    );
    assert!(JobError::Cancelled(String::new()).is_cancel());
    assert!(!JobError::Failed(String::new()).is_cancel());
}

// ── Stage plans ─────────────────────────────────────────────────────────────

#[test]
fn stages_split_the_range_by_weight() {
    let plan = StagePlan::new(&[("load", 1.0), ("align", 3.0)]);
    assert_eq!(plan.len(), 2);
    assert_eq!(plan.label(1), Some("align"));
    assert!(near(plan.fraction(0, 0.0), 0.0));
    assert!(near(plan.fraction(0, 1.0), 0.25));
    assert!(near(plan.fraction(1, 0.0), 0.25));
    assert!(near(plan.fraction(1, 0.5), 0.625));
    assert!(near(plan.fraction(1, 1.0), 1.0));
}

#[test]
fn a_zero_weight_stage_takes_no_share_and_out_of_range_input_is_clamped() {
    let plan = StagePlan::new(&[("load", 1.0), ("align", 0.0), ("fuse", 1.0)]);
    assert!(
        near(plan.fraction(1, 0.5), 0.5),
        "an empty stage is a point"
    );
    assert!(near(plan.fraction(0, 7.0), plan.fraction(0, 1.0)));
    assert!(near(plan.fraction(0, -3.0), 0.0));
    assert!(
        near(plan.fraction(9, 0.3), 1.0),
        "an unknown stage is the end"
    );
}

#[test]
fn all_zero_weights_share_the_range_equally() {
    let plan = StagePlan::new(&[("a", 0.0), ("b", 0.0)]);
    assert!(near(plan.fraction(0, 1.0), 0.5));
    assert!(near(plan.fraction(1, 1.0), 1.0));
    assert!(StagePlan::new(&[]).is_empty());
}

#[test]
fn fraction_of_counts_frames() {
    assert!(near(fraction_of(1, 4), 0.25));
    assert!(near(fraction_of(9, 4), 1.0));
    assert!(near(fraction_of(0, 0), 1.0));
}

// ── The monitor's queue ─────────────────────────────────────────────────────

fn files(names: &[&str]) -> Vec<PathBuf> {
    names.iter().map(PathBuf::from).collect()
}

#[test]
fn every_folder_change_is_one_queued_run_in_order() {
    let mut queue = RunQueue::default();
    assert!(queue.push(files(&["a.jpg"])));
    assert!(queue.push(files(&["a.jpg", "b.jpg"])));
    assert_eq!(queue.len(), 2);
    assert_eq!(queue.pop(), Some(files(&["a.jpg"])));
    assert_eq!(queue.pop(), Some(files(&["a.jpg", "b.jpg"])));
    assert_eq!(queue.pop(), None);
}

#[test]
fn a_repeated_scan_and_an_empty_scan_are_not_queued() {
    let mut queue = RunQueue::default();
    assert!(queue.push(files(&["a.jpg"])));
    assert!(
        !queue.push(files(&["a.jpg"])),
        "same list as the newest entry"
    );
    assert!(!queue.push(Vec::new()), "an empty folder is no run");
    assert_eq!(queue.len(), 1);
}

#[test]
fn stopping_the_monitor_drains_the_queue() {
    let mut queue = RunQueue::default();
    queue.push(files(&["a.jpg"]));
    queue.push(files(&["a.jpg", "b.jpg"]));
    assert_eq!(queue.drain(), 2);
    assert!(queue.is_empty());
    assert_eq!(queue.drain(), 0);
}

#[test]
fn the_monitor_chip_shows_the_queue() {
    assert_eq!(monitor_label(0, false), "Monitor: watching");
    assert_eq!(monitor_label(0, true), "Monitor: stacking");
    assert_eq!(monitor_label(2, true), "Monitor: 2 queued");
    assert_eq!(monitor_label(1, false), "Monitor: 1 queued");
}
