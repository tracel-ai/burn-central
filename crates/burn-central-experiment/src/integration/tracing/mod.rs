//! Tracing integration for experiment log forwarding.
//!
//! The default path is to enter an ambient experiment scope with
//! [`crate::ExperimentRun::in_scope`], [`crate::ExperimentHandle::in_scope`], or
//! [`crate::ExperimentInstrument::in_experiment`]. The tracing layer will forward events to the
//! current ambient experiment automatically.
//!
//! Use [`experiment_span`] when you want explicit span-based routing without ambient scope.

use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

use crate::ExperimentHandle;
use crate::integration::tracing::layer::ExperimentTracingLogLayer;

mod layer;
pub(crate) mod registry;
mod visitor;

/// Create a tracing layer that forwards tracing events to the active experiment in the current
/// tracing scope.
pub fn tracing_log_layer<S>() -> impl tracing_subscriber::Layer<S>
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    ExperimentTracingLogLayer
}

/// Best-effort initialization of a default tracing subscriber that includes experiment log
/// forwarding.
///
/// Returns `true` when a subscriber was installed and `false` when one was already installed or
/// initialization otherwise failed.
pub fn try_init_tracing_subscriber() -> bool {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_log_layer())
        .try_init()
        .is_ok()
}

/// Create a tracing span bound to the given experiment.
///
/// Events recorded within this span are routed to the experiment even when no ambient experiment
/// scope is active.
pub fn experiment_span(experiment: impl Into<ExperimentHandle>) -> tracing::Span {
    let experiment = experiment.into();
    tracing::info_span!("experiment", experiment_id = %experiment.id())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tracing_subscriber::layer::SubscriberExt;

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

        tracing::subscriber::with_default(
            subscriber,
            run.bind_current(|| {
                tracing::info!(step = 3u64, "epoch completed");
            }),
        );

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
