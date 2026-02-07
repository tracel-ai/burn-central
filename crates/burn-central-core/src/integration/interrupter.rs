use burn::train::Interrupter;

use crate::experiment::{Cancellable, ExperimentRun};

impl Cancellable for Interrupter {
    fn is_cancelled(&self) -> bool {
        self.should_stop()
    }

    fn cancel(&self) {
        self.stop(Some("Cancelled by user"));
    }
}

/// Helper for creating an `Interrupter` that is linked to the experiment's cancellation token.
/// When the experiment is cancelled, the `Interrupter` will be triggered, allowing any training loop that checks the `Interrupter` to stop gracefully.
pub fn remote_interrupter(experiment: &ExperimentRun) -> Interrupter {
    let cancel_token = experiment
        .cancel_token()
        .expect("Experiment should be in a valid state");
    let interrupter = Interrupter::new();
    cancel_token.link(interrupter.clone());
    interrupter
}
