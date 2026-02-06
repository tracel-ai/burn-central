use burn::train::Interrupter;

use crate::experiment::Cancellable;

impl Cancellable for Interrupter {
    fn is_cancelled(&self) -> bool {
        self.should_stop()
    }

    fn cancel(&self) {
        self.stop(Some("Cancelled by user"));
    }
}
