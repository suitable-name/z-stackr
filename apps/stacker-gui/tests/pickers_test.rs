use std::{
    cell::{Cell, RefCell},
    path::PathBuf,
    rc::Rc,
};

use stacker_gui::pickers::{
    PickerKind, PickerRequest, deliver, has_stubbed_answer, park, pick, stub_next_answer,
};

#[test]
fn a_stubbed_answer_is_delivered_exactly_once() {
    let calls = Rc::new(Cell::new(0_u32));
    let received: Rc<RefCell<Vec<PathBuf>>> = Rc::new(RefCell::new(Vec::new()));

    stub_next_answer(vec![PathBuf::from("chosen.png")]);
    assert!(has_stubbed_answer());

    let (calls_in, received_in) = (Rc::clone(&calls), Rc::clone(&received));
    pick(PickerRequest::new(PickerKind::OpenFile), move |paths| {
        calls_in.set(calls_in.get() + 1);
        *received_in.borrow_mut() = paths;
    });

    assert_eq!(calls.get(), 1, "the continuation ran once");
    assert_eq!(*received.borrow(), vec![PathBuf::from("chosen.png")]);
    assert!(
        !has_stubbed_answer(),
        "the stub is consumed by the first pick"
    );
}

#[test]
fn a_stubbed_cancel_delivers_an_empty_answer() {
    let seen: Rc<RefCell<Option<Vec<PathBuf>>>> = Rc::new(RefCell::new(None));

    stub_next_answer(Vec::new());
    let seen_in = Rc::clone(&seen);
    pick(PickerRequest::new(PickerKind::SaveFile), move |paths| {
        *seen_in.borrow_mut() = Some(paths);
    });

    assert_eq!(*seen.borrow(), Some(Vec::new()));
}

#[test]
fn a_parked_continuation_runs_once_per_key() {
    let calls = Rc::new(Cell::new(0_u32));
    let calls_in = Rc::clone(&calls);
    let key = park(move |paths| {
        assert_eq!(paths, vec![PathBuf::from("x.tif")]);
        calls_in.set(calls_in.get() + 1);
    });

    assert!(deliver(key, vec![PathBuf::from("x.tif")]));
    assert!(
        !deliver(key, vec![PathBuf::from("x.tif")]),
        "second delivery finds nothing"
    );
    assert_eq!(calls.get(), 1);
}

#[test]
fn parked_continuations_do_not_mix_up_their_answers() {
    let log: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let first_log = Rc::clone(&log);
    let second_log = Rc::clone(&log);
    let first = park(move |_| first_log.borrow_mut().push("first".to_owned()));
    let second = park(move |_| second_log.borrow_mut().push("second".to_owned()));
    assert_ne!(first, second);

    assert!(deliver(second, Vec::new()));
    assert!(deliver(first, Vec::new()));
    assert_eq!(*log.borrow(), vec!["second".to_owned(), "first".to_owned()]);
}

#[test]
fn request_builder_collects_everything_a_dialog_needs() {
    let request = PickerRequest::new(PickerKind::SaveFile)
        .titled("Save image")
        .filter("PNG Image", &["png"])
        .filter("TIFF Image", &["tif", "tiff"])
        .file_name("out.png")
        .starting_in(Some(PathBuf::from("pictures")));

    assert_eq!(request.kind, PickerKind::SaveFile);
    assert_eq!(request.title.as_deref(), Some("Save image"));
    assert_eq!(request.filters.len(), 2);
    assert_eq!(
        request.filters[1].extensions,
        vec!["tif".to_owned(), "tiff".to_owned()]
    );
    assert_eq!(request.default_file_name.as_deref(), Some("out.png"));
    assert_eq!(request.starting_dir, Some(PathBuf::from("pictures")));
}
