//! Backend-facing control plane for active experiment runs.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::activity::ActivityId;
use crate::cancellation::CancelToken;

/// Shared control plane for an active experiment run.
#[derive(Debug, Clone)]
pub struct ExperimentRunControl {
    inner: Arc<ControlInner>,
}

#[derive(Debug)]
struct ControlInner {
    run_cancel_token: CancelToken,
    activity_cancellations: Mutex<HashMap<ActivityId, CancelToken>>,
}

impl ExperimentRunControl {
    /// Create a control plane rooted at the provided run cancellation token.
    pub fn new(run_cancel_token: CancelToken) -> Self {
        Self {
            inner: Arc::new(ControlInner {
                run_cancel_token,
                activity_cancellations: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Return the run-level cancellation token.
    pub fn cancel_token(&self) -> CancelToken {
        self.inner.run_cancel_token.clone()
    }

    /// Request cancellation of the whole experiment run.
    pub fn cancel_run(&self) {
        self.inner.run_cancel_token.cancel();
    }

    /// Return whether cancellation has been requested for the whole run.
    pub fn is_run_cancelled(&self) -> bool {
        self.inner.run_cancel_token.is_cancelled()
    }

    /// Request cancellation of a registered cancellable activity.
    ///
    /// Returns `true` when an activity token was found and cancellation was requested.
    pub fn cancel_activity(&self, id: ActivityId) -> bool {
        let token = self
            .inner
            .activity_cancellations
            .lock()
            .unwrap()
            .get(&id)
            .cloned();

        if let Some(token) = token {
            token.cancel();
            true
        } else {
            false
        }
    }

    pub(crate) fn register_activity_cancellation(&self, id: ActivityId, token: CancelToken) {
        self.inner
            .activity_cancellations
            .lock()
            .unwrap()
            .insert(id, token);
    }

    pub(crate) fn unregister_activity_cancellation(&self, id: ActivityId) {
        self.inner
            .activity_cancellations
            .lock()
            .unwrap()
            .remove(&id);
    }
}

impl Default for ExperimentRunControl {
    fn default() -> Self {
        Self::new(CancelToken::new())
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;

    fn activity_id(value: u64) -> ActivityId {
        ActivityId::new(NonZeroU64::new(value).unwrap())
    }

    #[test]
    fn cancel_run_cancels_linked_activity_tokens() {
        let control = ExperimentRunControl::default();
        let activity_token = control.cancel_token().linked(CancelToken::new());
        control.register_activity_cancellation(activity_id(1), activity_token.clone());

        control.cancel_run();

        assert!(control.is_run_cancelled());
        assert!(activity_token.is_cancelled());
    }

    #[test]
    fn cancel_activity_cancels_registered_activity_token() {
        let control = ExperimentRunControl::default();
        let activity_token = CancelToken::new();
        control.register_activity_cancellation(activity_id(1), activity_token.clone());

        assert!(control.cancel_activity(activity_id(1)));

        assert!(activity_token.is_cancelled());
    }

    #[test]
    fn cancel_unknown_activity_returns_false() {
        let control = ExperimentRunControl::default();

        assert!(!control.cancel_activity(activity_id(99)));
    }
}
