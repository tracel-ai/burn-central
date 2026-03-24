use std::ops::Deref;
use std::sync::{Arc, Mutex, Weak};

use burn_central_artifact::bundle::{BundleDecode, BundleEncode, FsBundle};
use burn_central_client::Client;
use serde::Serialize;

mod cancellation;
pub mod error;
pub mod integration;
mod reader;
mod remote;
mod session;

pub use cancellation::{CancelToken, Cancellable};
pub use session::ExperimentCompletion;

use crate::error::{ExperimentError, ExperimentErrorKind};
use crate::reader::ExperimentArtifactReader;
use crate::session::{Event, ExperimentSession};

/// The unique identifier for an experiment run. It is used to associate events, artifacts, and other data with a specific experiment.
pub type ExperimentId = String;

/// The different types of artifacts that can be associated with an experiment run. It is used to categorize artifacts and provide context for their usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    Model,
    Log,
    Other,
}

/// The data needed for a metric that can be logged during an experiment run.
#[derive(Debug, Clone)]
pub struct MetricSpec {
    pub name: String,
    pub description: Option<String>,
    pub unit: Option<String>,
    pub higher_is_better: bool,
}

/// The value of a metric at a specific point during an experiment run.
#[derive(Debug, Clone)]
pub struct MetricValue {
    pub name: String,
    pub value: f64,
}

#[derive(Debug, Clone)]
struct ExperimentMetadata {
    pub id: ExperimentId,
}

/// The main struct representing an active experiment run. It provides methods for logging events, saving artifacts, and managing the lifecycle of the experiment.
/// When dropped, it will automatically mark the experiment as finished, ensuring that resources are cleaned up properly.
pub struct ExperimentRun {
    inner: Arc<RunInner>,
    handle: ExperimentHandle,
}

/// The handle for an experiment run, which can be cloned and used to interact with the experiment from different parts of the code.
/// It provides methods for logging events and saving artifacts, and it ensures that the underlying experiment run is still active before performing any operations.
#[derive(Clone)]
pub struct ExperimentHandle {
    metadata: ExperimentMetadata,
    inner: Weak<RunInner>,
}

struct RunInner {
    metadata: ExperimentMetadata,
    cancel_token: CancelToken,
    state: Mutex<RunState>,
    session: Box<dyn ExperimentSession>,
    reader: Box<dyn ExperimentArtifactReader>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunState {
    Active,
    Finished,
}

impl ExperimentRun {
    pub fn new<S, R>(
        id: impl Into<ExperimentId>,
        session: S,
        reader: R,
        cancel_token: CancelToken,
    ) -> Self
    where
        S: ExperimentSession + 'static,
        R: ExperimentArtifactReader + 'static,
    {
        let metadata = ExperimentMetadata { id: id.into() };
        let inner = Arc::new(RunInner {
            metadata: metadata.clone(),
            cancel_token: cancel_token.clone(),
            state: Mutex::new(RunState::Active),
            session: Box::new(session),
            reader: Box::new(reader),
        });

        let handle = ExperimentHandle {
            metadata,
            inner: Arc::downgrade(&inner),
        };

        Self { inner, handle }
    }

    /// Get a handle to the experiment run that can be used to access experiment capabilities in a thread-safe way.
    pub fn handle(&self) -> ExperimentHandle {
        self.handle.clone()
    }

    /// Get the unique identifier for this experiment run.
    pub fn id(&self) -> &ExperimentId {
        &self.inner.metadata.id
    }

    /// Get a cancellation token that can be used to link child tasks to be cancelled when this experiment run is cancelled.
    pub fn cancel_token(&self) -> CancelToken {
        self.inner.cancel_token.clone()
    }

    /// Cancel the experiment run, marking it as cancelled and preventing any further events or artifacts from being recorded.
    pub fn cancel(&self) -> Result<(), ExperimentError> {
        self.inner.ensure_active()?;
        self.inner.cancel_token.cancel();
        Ok(())
    }

    /// Mark the experiment run as finished with the given completion status. This will prevent any further events or artifacts from being recorded and will trigger any necessary cleanup in the backend.
    pub fn finish(self) -> Result<(), ExperimentError> {
        self.inner.finish_once(ExperimentCompletion::Success)
    }

    /// Mark the experiment run as failed with the given reason. This will prevent any further events or artifacts from being recorded and will trigger any necessary cleanup in the backend.
    pub fn fail(self, reason: impl Into<String>) -> Result<(), ExperimentError> {
        self.inner
            .finish_once(ExperimentCompletion::Failed(reason.into()))
    }
}

/// Provides methods for the remote experiment run implementation using Burn Central.
impl ExperimentRun {
    pub fn remote(
        client: Client,
        namespace: &str,
        project_name: &str,
        digest: String,
        routine: String,
    ) -> Result<Self, ExperimentError> {
        remote::create_experiment_run(client, namespace, project_name, digest, routine).map_err(
            |e| {
                ExperimentError::with_source(
                    ExperimentErrorKind::Internal,
                    "Failed to start remote experiment run",
                    e,
                )
            },
        )
    }
}

/// Provides methods for the local experiment run implementation without Burn Central.
impl ExperimentRun {
    pub fn local() -> Self {
        todo!("Local experiment run is not yet implemented")
    }
}

impl Deref for ExperimentRun {
    type Target = ExperimentHandle;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl ExperimentHandle {
    pub fn id(&self) -> &ExperimentId {
        &self.metadata.id
    }

    fn record_event(&self, event: Event) -> Result<(), ExperimentError> {
        let inner = self.upgrade()?;
        inner.ensure_active()?;
        inner.session.record_event(event)
    }

    pub fn log_args<A: Serialize>(&self, args: &A) -> Result<(), ExperimentError> {
        let value = serde_json::to_value(args).map_err(|e| {
            ExperimentError::with_source(
                ExperimentErrorKind::Artifact,
                "Failed to serialize experiment arguments",
                e,
            )
        })?;

        self.record_event(Event::Args(value))
    }

    pub fn log_config<C: Serialize>(
        &self,
        name: impl Into<String>,
        config: &C,
    ) -> Result<(), ExperimentError> {
        let value = serde_json::to_value(config).map_err(|e| {
            ExperimentError::with_source(
                ExperimentErrorKind::Artifact,
                "Failed to serialize experiment config",
                e,
            )
        })?;

        self.record_event(Event::Config {
            name: name.into(),
            value,
        })
    }

    pub fn log_info(&self, message: impl Into<String>) -> Result<(), ExperimentError> {
        self.record_event(Event::Log {
            message: message.into(),
        })
    }

    pub fn log_metric(
        &self,
        epoch: usize,
        split: impl Into<String>,
        iteration: usize,
        items: Vec<MetricValue>,
    ) -> Result<(), ExperimentError> {
        self.record_event(Event::Metrics {
            epoch,
            split: split.into(),
            iteration,
            items,
        })
    }

    pub fn log_metric_definition(&self, spec: MetricSpec) -> Result<(), ExperimentError> {
        self.record_event(Event::MetricDefinition(spec))
    }

    pub fn log_epoch_summary(
        &self,
        epoch: usize,
        split: impl Into<String>,
        items: Vec<MetricValue>,
    ) -> Result<(), ExperimentError> {
        self.record_event(Event::EpochSummary {
            epoch,
            split: split.into(),
            items,
        })
    }

    pub fn save_artifact<E: BundleEncode>(
        &self,
        name: impl AsRef<str>,
        kind: ArtifactKind,
        artifact: E,
        settings: &E::Settings,
    ) -> Result<(), ExperimentError> {
        let inner = self.upgrade()?;
        inner.ensure_active()?;

        let mut bundle = FsBundle::temp().map_err(|e| {
            ExperimentError::with_source(
                ExperimentErrorKind::Artifact,
                "Failed to create temporary bundle for artifact",
                e,
            )
        })?;

        artifact.encode(&mut bundle, settings).map_err(|e| {
            ExperimentError::with_source(
                ExperimentErrorKind::Artifact,
                "Failed to encode artifact into bundle",
                e,
            )
        })?;

        inner.session.save_artifact(name.as_ref(), kind, &bundle)
    }

    pub fn use_artifact<D: BundleDecode>(
        &self,
        experiment_id: impl Into<ExperimentId>,
        name: impl AsRef<str>,
        settings: &D::Settings,
    ) -> Result<D, ExperimentError> {
        let inner = self.upgrade()?;
        inner.ensure_active()?;
        let artifact = inner
            .reader
            .load_artifact_raw(experiment_id.into(), name.as_ref())
            .map_err(|e| {
                ExperimentError::with_source(
                    ExperimentErrorKind::Artifact,
                    format!("Failed to load artifact bundle for {}", name.as_ref()),
                    e,
                )
            })?;

        D::decode(&artifact.bundle, settings).map_err(|e| {
            ExperimentError::with_source(
                ExperimentErrorKind::Artifact,
                format!("Failed to decode artifact: {}", name.as_ref()),
                e,
            )
        })
    }

    fn upgrade(&self) -> Result<Arc<RunInner>, ExperimentError> {
        self.inner.upgrade().ok_or(ExperimentError::new(
            ExperimentErrorKind::InactiveRun,
            "Experiment run is no longer active",
        ))
    }
}

impl RunInner {
    fn ensure_active(&self) -> Result<(), ExperimentError> {
        if self.cancel_token.is_cancelled() {
            return Err(ExperimentError::new(
                ExperimentErrorKind::Cancelled,
                "Experiment run has been cancelled",
            ));
        }

        let state = self.state.lock().unwrap();
        match *state {
            RunState::Active => Ok(()),
            RunState::Finished => Err(ExperimentError::new(
                ExperimentErrorKind::AlreadyFinished,
                "Experiment run has already finished",
            )),
        }
    }

    fn finish_once(&self, completion: ExperimentCompletion) -> Result<(), ExperimentError> {
        let mut state = self.state.lock().unwrap();
        match *state {
            RunState::Finished => Err(ExperimentError::new(
                ExperimentErrorKind::AlreadyFinished,
                "Experiment run has already finished",
            )),
            RunState::Active => {
                *state = RunState::Finished;
                drop(state);
                self.session.finish(completion)
            }
        }
    }
}

/// The experiment run is marked as finished when the `ExperimentRun` struct is dropped. If the run was not explicitly finished or cancelled, it will be marked as successful by default.
impl Drop for ExperimentRun {
    fn drop(&mut self) {
        let completion = if self.inner.cancel_token.is_cancelled() {
            ExperimentCompletion::Cancelled
        } else {
            ExperimentCompletion::Success
        };

        let _ = self.inner.finish_once(completion);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::reader::{ExperimentReaderError, LoadedArtifact};

    use super::*;

    #[derive(Default)]
    struct MockSession {
        events: Mutex<Vec<Event>>,
        completions: Mutex<Vec<ExperimentCompletion>>,
        artifacts_saved: AtomicUsize,
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
            _artifact: &FsBundle,
        ) -> Result<(), ExperimentError> {
            self.artifacts_saved.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }

        fn finish(&self, completion: ExperimentCompletion) -> Result<(), ExperimentError> {
            self.completions.lock().unwrap().push(completion);
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

    fn create_run(session: Arc<MockSession>) -> ExperimentRun {
        ExperimentRun::new(
            "test/experiment/1",
            session,
            NoopExperimentDataReader,
            CancelToken::default(),
        )
    }

    #[test]
    fn run_derefs_to_handle_for_event_recording() {
        let session = Arc::new(MockSession::default());
        let run = create_run(session.clone());

        run.log_info("hello").unwrap();

        let events = session.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::Log { message } => assert_eq!(message, "hello"),
            event => panic!("unexpected event: {event:?}"),
        }
    }

    #[test]
    fn finish_marks_handle_inactive() {
        let session = Arc::new(MockSession::default());
        let run = create_run(session.clone());
        let handle = run.handle();

        run.finish().unwrap();

        let err = handle.log_info("after-finish").unwrap_err();
        assert_eq!(err.kind, ExperimentErrorKind::InactiveRun);
    }

    #[test]
    fn drop_marks_run_as_finished_successfully() {
        let session = Arc::new(MockSession::default());

        {
            let _run = create_run(session.clone());
        }

        let completions = session.completions.lock().unwrap();
        assert_eq!(completions.as_slice(), &[ExperimentCompletion::Success]);
    }

    #[test]
    fn cancel_marks_run_cancelled_on_drop() {
        let session = Arc::new(MockSession::default());

        {
            let run = create_run(session.clone());
            run.cancel().unwrap();
        }

        let completions = session.completions.lock().unwrap();
        assert_eq!(completions.as_slice(), &[ExperimentCompletion::Cancelled]);
    }
}
