use burn::train::Interrupter;

use crate::{Cancellable, ExperimentRun};

struct LinkedInterrupter(Interrupter);

impl Cancellable for LinkedInterrupter {
    fn is_cancelled(&self) -> bool {
        self.0.should_stop()
    }

    fn cancel(&self) {
        self.0.stop(Some("Cancelled by user"));
    }
}

/// Helper for creating an `Interrupter` that is linked to the experiment's cancellation token.
/// When the experiment is cancelled, the `Interrupter` will be triggered, allowing any training loop that checks the `Interrupter` to stop gracefully.
pub fn remote_interrupter(experiment: &ExperimentRun) -> Interrupter {
    let cancel_token = experiment.cancel_token();
    let interrupter = Interrupter::new();
    cancel_token.link(LinkedInterrupter(interrupter.clone()));
    interrupter
}
