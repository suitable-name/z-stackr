//! UI flow tests: the real window and the real Rust wiring, driven by clicks and
//! key presses on Slint's headless testing backend.
//!
//! The testing backend can be installed once per process and runs one event
//! loop, so every scenario runs inside the single `#[test]` below, one after the
//! other, each on a freshly booted app. A failing scenario does not stop the
//! others; the test fails at the end with the list of every broken scenario.

// The scenarios are futures on Slint's single-threaded event loop; they hold
// window handles across awaits and are never sent to another thread.
#![expect(clippy::future_not_send, reason = "single-threaded Slint event loop")]

mod flows;
mod harness;
mod matrix;
mod scenarios;

use harness::Outcome;

type ScenarioFn = (
    &'static str,
    fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Outcome>>>,
);

#[test]
fn ui_flows() {
    i_slint_backend_testing::init_integration_test_with_system_time();
    let filter = std::env::var("UI_FLOW").ok();
    let failures = std::rc::Rc::new(std::cell::RefCell::new(Vec::<String>::new()));
    let ran = std::rc::Rc::new(std::cell::Cell::new(0_usize));
    {
        let failures = std::rc::Rc::clone(&failures);
        let ran = std::rc::Rc::clone(&ran);
        slint::spawn_local(async move {
            for (name, run) in scenarios::ALL {
                if filter.as_deref().is_some_and(|f| !name.contains(f)) {
                    continue;
                }
                ran.set(ran.get() + 1);
                eprintln!("ui flow: {name}");
                if let Err(message) = run().await {
                    eprintln!("ui flow: {name} FAILED: {message}");
                    failures.borrow_mut().push(format!("{name}: {message}"));
                }
            }
            slint::quit_event_loop().unwrap();
        })
        .unwrap();
    }
    slint::run_event_loop_until_quit().unwrap();
    let failures = failures.borrow();
    assert!(ran.get() > 0, "no scenario ran");
    assert!(
        failures.is_empty(),
        "{} of {} UI flows failed:\n{}",
        failures.len(),
        ran.get(),
        failures.join("\n")
    );
}
