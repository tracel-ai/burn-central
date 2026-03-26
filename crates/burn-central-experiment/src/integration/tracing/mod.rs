//! Tracing integration for routing `tracing` events into experiments.
//!
//! Install [`tracing_log_layer`] or call [`try_init_tracing_subscriber`] to enable forwarding.
//! Once installed, you can choose between two routing styles:
//! - ambient routing with [`crate::ExperimentGlobalExt::enter`],
//!   [`crate::ExperimentGlobalExt::in_scope`], or
//!   [`crate::ExperimentInstrument::in_experiment`]
//! - explicit span routing with [`ExperimentTracingExt::tracing_span`]
//!
//! Ambient routing is usually the simplest option when your code already has access to an
//! [`crate::ExperimentRun`] or [`crate::ExperimentRunHandle`]. Span routing is useful when you
//! need to bind logs to an experiment without relying on thread-local ambient state.

use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

use crate::{ExperimentRun, ExperimentRunHandle};

mod layer;
pub(crate) mod registry;
mod visitor;

pub use layer::ExperimentTracingLogLayer;

/// Create a layer that forwards `tracing` events to the experiment associated with the current
/// tracing scope.
///
/// This is a convenience constructor for [`ExperimentTracingLogLayer`].
pub fn tracing_log_layer() -> ExperimentTracingLogLayer {
    ExperimentTracingLogLayer
}

/// Best-effort initialization of a default tracing subscriber that includes experiment log
/// forwarding.
///
/// Returns `true` when a subscriber was installed and `false` when one was already installed or
/// initialization otherwise failed.
///
/// This is a convenience for binaries that do not already configure their own subscriber.
pub fn try_init_tracing_subscriber() -> bool {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_log_layer())
        .try_init()
        .is_ok()
}

/// Build an explicit tracing span tied to a specific experiment.
fn experiment_span(experiment: impl Into<ExperimentRunHandle>) -> tracing::Span {
    let experiment = experiment.into();
    tracing::info_span!("experiment", experiment_id = %experiment.id())
}

/// Extension trait for creating explicit experiment-bound tracing spans.
///
/// Implemented for both [`ExperimentRun`] and [`ExperimentRunHandle`].
pub trait ExperimentTracingExt {
    /// Create a tracing span bound to this experiment.
    ///
    /// Use this when ambient routing through [`crate::ExperimentGlobalExt`] is not practical or
    /// when you want the routing to be explicit in the span tree.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use burn_central_experiment::ExperimentRun;
    /// use burn_central_experiment::integration::tracing::{ExperimentTracingExt};
    ///
    /// # fn main() {
    /// let experiment = ExperimentRun::local("/tmp/experiments").unwrap();
    /// let span: tracing::Span = experiment.tracing_span();
    /// let _guard = span.enter();
    /// tracing::info!("this event is routed to the experiment");
    /// # }
    /// ```
    #[must_use = "span must be entered to route events"]
    fn tracing_span(&self) -> tracing::Span;
}

impl ExperimentTracingExt for ExperimentRun {
    fn tracing_span(&self) -> tracing::Span {
        experiment_span(self)
    }
}

impl ExperimentTracingExt for ExperimentRunHandle {
    fn tracing_span(&self) -> tracing::Span {
        experiment_span(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tracing_subscriber::layer::SubscriberExt;

    use crate::context::ExperimentGlobalExt;
    use crate::error::ExperimentError;
    use crate::reader::{ExperimentArtifactReader, ExperimentReaderError, LoadedArtifact};
    use crate::session::{BundleFn, Event, ExperimentCompletion, ExperimentSession};
    use crate::{ArtifactKind, CancelToken, ExperimentId, ExperimentRun};

    use super::*;

    #[derive(Default)]
    struct MockSession {
        events: Mutex<Vec<Event>>,
    }

    impl ExperimentSession for MockSession {
        fn record_event(&self, event: Event) -> Result<(), ExperimentError> {
            self.events.lock().unwrap().push(event);
            Ok(())
        }

        fn save_artifact(
            &self,
            _name: &str,
            _kind: ArtifactKind,
            _artifact: Box<BundleFn>,
        ) -> Result<(), ExperimentError> {
            Ok(())
        }

        fn finish(&self, _completion: ExperimentCompletion) -> Result<(), ExperimentError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct NoopExperimentDataReader;

    impl ExperimentArtifactReader for NoopExperimentDataReader {
        fn load_artifact_raw(
            &self,
            _experiment_id: ExperimentId,
            _name: &str,
        ) -> Result<LoadedArtifact, ExperimentReaderError> {
            Err(ExperimentReaderError::new("Artifact not found"))
        }
    }

    fn create_run(id: &str, session: Arc<MockSession>) -> ExperimentRun {
        ExperimentRun::new(
            id,
            session,
            NoopExperimentDataReader,
            CancelToken::default(),
        )
    }

    #[test]
    fn tracing_layer_forwards_events_to_current_experiment() {
        let session = Arc::new(MockSession::default());
        let run = create_run("trace-test-1", session.clone());
        let subscriber = tracing_subscriber::registry().with(tracing_log_layer());

        tracing::subscriber::with_default(subscriber, || {
            run.in_scope(|| {
                tracing::info!(step = 3u64, "epoch completed");
            })
        });

        let events = session.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::Log { message } => {
                assert!(message.contains("epoch completed"));
                assert!(message.contains("step=3"));
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }

    #[test]
    fn tracing_layer_routes_from_span_experiment_id_without_ambient_scope() {
        let session = Arc::new(MockSession::default());
        let run = create_run("trace-test-span", session.clone());
        let subscriber = tracing_subscriber::registry().with(tracing_log_layer());

        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!("experiment", experiment_id = %run.id());
            let _guard = span.enter();
            tracing::info!(step = 7u64, "span-routed event");
        });

        let events = session.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::Log { message } => {
                assert!(message.contains("span-routed event"));
                assert!(message.contains("step=7"));
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }

    #[test]
    fn tracing_layer_routes_from_experiment_span_helper_without_ambient_scope() {
        let session = Arc::new(MockSession::default());
        let run = create_run("trace-test-helper-span", session.clone());
        let subscriber = tracing_subscriber::registry().with(tracing_log_layer());

        tracing::subscriber::with_default(subscriber, || {
            let span = experiment_span(&run);
            let _guard = span.enter();
            tracing::info!("helper-span-routed event");
        });

        let events = session.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::Log { message } => {
                assert!(message.contains("helper-span-routed event"));
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }

    #[test]
    fn tracing_layer_skips_events_without_experiment_scope() {
        let session = Arc::new(MockSession::default());
        let _run = create_run("trace-test-2", session.clone());
        let subscriber = tracing_subscriber::registry().with(tracing_log_layer());

        tracing::subscriber::with_default(subscriber, || {
            tracing::info!("outside experiment scope");
        });

        let events = session.events.lock().unwrap();
        assert!(events.is_empty());
    }
}
