use burn::train::Interrupter;

use crate::{Cancellable, ExperimentRunHandle};

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
///
/// Prefer [`crate::integration::training::ExperimentTrainingExt::interrupter`] when you already
/// have an [`ExperimentRun`][crate::ExperimentRun] in scope.
pub fn experiment_interrupter(experiment: impl Into<ExperimentRunHandle>) -> Interrupter {
    let cancel_token = experiment.into().cancel_token();
    let interrupter = Interrupter::new();
    cancel_token.link(LinkedInterrupter(interrupter.clone()));
    interrupter
}
