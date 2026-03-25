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

/// Create an [`Interrupter`] linked to an experiment run's cancellation token.
///
/// When the run is cancelled, the returned interrupter will request a graceful stop from any
/// training loop that checks it.
pub fn remote_interrupter(experiment: &ExperimentRun) -> Interrupter {
    let cancel_token = experiment.cancel_token();
    let interrupter = Interrupter::new();
    cancel_token.link(LinkedInterrupter(interrupter.clone()));
    interrupter
}
