//! Backend-agnostic experiment tracking primitives.
//!
//! [`ExperimentRun`] is the active run abstraction used by both the Burn Central backend and the
//! local backend. Use [`ExperimentRun::remote`] to create a run backed by Burn Central, or
//! [`ExperimentRun::local`] to create numbered local runs under a user-provided root directory.
//!
//! The run dereferences to an [`ExperimentHandle`] for logging events and saving artifacts. Burn
//! learner integrations built on top of this API are available in [`integration`].

use std::fmt;
use std::ops::Deref;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex, Weak};

use burn_central_artifact::bundle::{BundleDecode, BundleEncode, FsBundle};
use burn_central_client::Client;
use serde::Serialize;

mod cancellation;
pub mod error;
pub mod integration;
mod local;
mod reader;
mod remote;
mod session;

pub use cancellation::{CancelToken, Cancellable};
pub use session::ExperimentCompletion;

use crate::error::{ExperimentError, ExperimentErrorKind};
use crate::reader::ExperimentArtifactReader;
use crate::session::{Event, ExperimentSession};

/// Opaque identifier for an experiment run.
///
/// The identifier format is backend-specific. It is stable for the backend that created it, but it
/// should not be interpreted across different backends.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExperimentId(String);

impl ExperimentId {
    /// Create an experiment identifier from a backend-specific string value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the backend-specific identifier value.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Try to parse the identifier value into another type.
    pub fn parse<T: FromStr>(&self) -> Option<T> {
        self.0.parse().ok()
    }
}

impl fmt::Display for ExperimentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ExperimentId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ExperimentId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<&String> for ExperimentId {
    fn from(value: &String) -> Self {
        Self(value.clone())
    }
}

impl From<i32> for ExperimentId {
    fn from(value: i32) -> Self {
        Self(value.to_string())
    }
}

impl From<u32> for ExperimentId {
    fn from(value: u32) -> Self {
        Self(value.to_string())
    }
}

/// Artifact category associated with an experiment run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    /// Model weights, parameters, or checkpoints.
    Model,
    /// Log files or related textual outputs.
    Log,
    /// Any artifact that does not fit a more specific category.
    Other,
}

/// Metric definition metadata logged during a run.
#[derive(Debug, Clone)]
pub struct MetricSpec {
    /// Display name for the metric.
    pub name: String,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Optional unit associated with the metric value.
    pub unit: Option<String>,
    /// Whether higher values are considered better.
    pub higher_is_better: bool,
}

/// Numeric metric value logged during a run.
#[derive(Debug, Clone)]
pub struct MetricValue {
    /// Metric name.
    pub name: String,
    /// Metric value.
    pub value: f64,
}

#[derive(Debug, Clone)]
struct ExperimentMetadata {
    pub id: ExperimentId,
}

/// Active experiment run.
///
/// This type owns the run lifecycle. It dereferences to [`ExperimentHandle`] for logging events
/// and saving artifacts. If dropped without an explicit completion, it is finalized automatically.
pub struct ExperimentRun {
    inner: Arc<RunInner>,
    handle: ExperimentHandle,
}

/// Cloneable handle for interacting with an active experiment run.
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
    /// Create an experiment run from backend-provided session and reader implementations.
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

    /// Clone a handle that can be shared across tasks and threads.
    pub fn handle(&self) -> ExperimentHandle {
        self.handle.clone()
    }

    /// Borrow the identifier for this run.
    pub fn id(&self) -> &ExperimentId {
        &self.inner.metadata.id
    }

    /// Get a token that can be linked to child tasks.
    pub fn cancel_token(&self) -> CancelToken {
        self.inner.cancel_token.clone()
    }

    /// Signal that the experiment run has been cancelled.
    ///
    /// The run remains usable until it is explicitly finished, failed, or dropped. If it is later
    /// dropped without an explicit completion, it will be marked as cancelled.
    pub fn cancel(&self) -> Result<(), ExperimentError> {
        self.inner.ensure_active()?;
        self.inner.cancel_token.cancel();
        Ok(())
    }

    /// Mark the run as successful and finalize the backend session.
    ///
    /// If the run is dropped without calling [`Self::finish`] or [`Self::fail`], it is finalized
    /// as successful by default.
    pub fn finish(self) -> Result<(), ExperimentError> {
        self.inner.finish_once(ExperimentCompletion::Success)
    }

    /// Mark the run as failed and finalize the backend session.
    pub fn fail(self, reason: impl Into<String>) -> Result<(), ExperimentError> {
        self.inner
            .finish_once(ExperimentCompletion::Failed(reason.into()))
    }
}

impl ExperimentRun {
    /// Create a run backed by Burn Central.
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

impl ExperimentRun {
    /// Create a local run under the provided root directory.
    ///
    /// Each call creates a numbered run directory inside `root`.
    pub fn local(root: impl Into<PathBuf>) -> Result<Self, ExperimentError> {
        local::create_experiment_run(root.into())
    }
}

impl Deref for ExperimentRun {
    type Target = ExperimentHandle;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl ExperimentHandle {
    /// Borrow the identifier for the underlying run.
    pub fn id(&self) -> &ExperimentId {
        &self.metadata.id
    }

    fn record_event(&self, event: Event) -> Result<(), ExperimentError> {
        let inner = self.upgrade()?;
        inner.ensure_active()?;
        inner.session.record_event(event)
    }

    /// Log the serialized input arguments for the run.
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

    /// Log a named configuration object for the run.
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

    /// Log an informational message for the run.
    pub fn log_info(&self, message: impl Into<String>) -> Result<(), ExperimentError> {
        self.record_event(Event::Log {
            message: message.into(),
        })
    }

    /// Log metric values for an epoch, split, and iteration.
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

    /// Log a metric definition so later metric values have metadata attached.
    pub fn log_metric_definition(&self, spec: MetricSpec) -> Result<(), ExperimentError> {
        self.record_event(Event::MetricDefinition(spec))
    }

    /// Log aggregated metric values for an epoch and split.
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

    /// Encode and persist an artifact in the configured backend.
    pub fn save_artifact<E: BundleEncode>(
        &self,
        name: impl AsRef<str>,
        kind: ArtifactKind,
        artifact: E,
        settings: &E::Settings,
    ) -> Result<(), ExperimentError> {
        let inner = self.upgrade()?;
        inner.ensure_active()?;

        let artifact_fn = |bundle: &mut FsBundle| {
            artifact.encode(bundle, settings).map_err(|e| {
                ExperimentError::with_source(
                    ExperimentErrorKind::Artifact,
                    "Failed to encode artifact into bundle",
                    e,
                )
            })
        };

        inner
            .session
            .save_artifact(name.as_ref(), kind, Box::new(artifact_fn))
    }

    /// Load and decode an artifact from a compatible experiment identifier.
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

/// Finalize the run on drop if it has not already been completed.
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
    use std::io::Read;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::reader::{ExperimentReaderError, LoadedArtifact};
    use crate::session::BundleFn;
    use burn_central_artifact::bundle::{BundleSink, BundleSource};

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
            _artifact: Box<BundleFn>,
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

    #[test]
    fn cancel_does_not_prevent_logging_before_drop() {
        let session = Arc::new(MockSession::default());
        let run = create_run(session.clone());

        run.cancel().unwrap();
        run.log_info("still-logging").unwrap();

        let events = session.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::Log { message } => assert_eq!(message, "still-logging"),
            event => panic!("unexpected event: {event:?}"),
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestArtifact(String);

    impl BundleEncode for TestArtifact {
        type Settings = ();
        type Error = String;

        fn encode<O: BundleSink>(
            self,
            sink: &mut O,
            _settings: &Self::Settings,
        ) -> Result<(), Self::Error> {
            sink.put_bytes("artifact.txt", self.0.as_bytes())
        }
    }

    impl BundleDecode for TestArtifact {
        type Settings = ();
        type Error = String;

        fn decode<I: BundleSource>(
            source: &I,
            _settings: &Self::Settings,
        ) -> Result<Self, Self::Error> {
            let mut reader = source.open("artifact.txt")?;
            let mut content = String::new();
            reader
                .read_to_string(&mut content)
                .map_err(|err| err.to_string())?;
            Ok(Self(content))
        }
    }

    #[test]
    fn local_run_roundtrips_saved_artifacts() {
        let run_root = std::env::temp_dir().join(format!(
            "burn-central-experiment-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let run = ExperimentRun::local(run_root).unwrap();
        let run_id = run.id().clone();

        run.save_artifact(
            "artifact",
            ArtifactKind::Other,
            TestArtifact("hello".into()),
            &(),
        )
        .unwrap();

        let loaded = run
            .use_artifact::<TestArtifact>(run_id, "artifact", &())
            .unwrap();

        assert_eq!(loaded, TestArtifact("hello".into()));
    }
}
