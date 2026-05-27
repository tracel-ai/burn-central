//! Experiment tracking primitives.
//!
//! This crate revolves around two core types:
//! - [`ExperimentRun`], which owns the lifecycle of an active experiment.
//! - [`ExperimentRunHandle`], which is a lightweight cloneable view for logging and artifact access
//!   from background tasks or other threads.
//!
//! Optional capabilities are exposed through extension traits:
//! - [`ExperimentRunHandleExt`] for cloning a shareable handle.
//! - [`ExperimentGlobalExt`] for ambient thread-local experiment context.
//! - [`integration::training::ExperimentTrainingExt`] for Burn `train` adapters.
//! - [`integration::tracing::ExperimentTracingExt`] for tracing span helpers.
//!
//! Backends are connected through the [`ExperimentProvider`] port. [`ExperimentModule`] and
//! [`ExperimentJob`] are the user-facing entry points for running a job and logging its result.

use std::fmt;
use std::str::FromStr;
use std::sync::{Arc, Mutex, Weak};

use tracel_artifact::bundle::{BundleDecode, BundleEncode, FsBundle};

use serde::Serialize;

mod activity;
mod cancellation;
mod context;
mod provider;
pub mod reader;
pub mod session;

pub mod error;
pub mod integration;

pub use activity::{ActivityBuilder, ActivityGuard, Metered, Unmetered};
pub use cancellation::{CancelToken, Cancellable};
pub use context::{
    CurrentExperimentGuard, ExperimentGlobalExt, ExperimentInstrument, WithCurrentExperiment,
};

use crate::activity::AtomicActivityIdAllocator;
use crate::error::{ExperimentError, ExperimentErrorKind};
use crate::integration::tracing::registry::{TracingRegistration, TracingRegistry};
use crate::reader::ExperimentArtifactReader;
use crate::session::{Event, ExperimentCompletion, ExperimentSession};
pub use cancellation::{CancelToken, Cancellable};
pub use context::{
    CurrentExperimentGuard, ExperimentGlobalExt, ExperimentInstrument, WithCurrentExperiment,
};
pub use progress::{ProgressBuilder, ProgressGuard};
pub use provider::{ExperimentFn, ExperimentJob, ExperimentModule, ExperimentProvider};

/// Opaque identifier for an experiment run.
///
/// The identifier format is backend-specific.
///
/// It is stable for the backend that created it, but it should not be interpreted across different backends.
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

/// An active experiment run.
///
/// `ExperimentRun` owns finalization. As long as the run remains active, it can log structured
/// events, persist artifacts, and expose a cancellation token to child work.
///
/// Use [`ExperimentRunHandleExt::handle`] when you need to share logging and artifact access without
/// transferring lifecycle ownership. If the run is dropped without an explicit completion, it is
/// finalized automatically.
///
/// Use [`ExperimentGlobalExt`] when you want to make the run available as the ambient
/// thread-local experiment for tracing or other integrations.
pub struct ExperimentRun {
    inner: Arc<RunInner>,
    handle: ExperimentRunHandle,
    _tracing_registration: TracingRegistration,
}

/// Cloneable handle for interacting with an active experiment run.
///
/// A handle keeps the run identifier plus logging and artifact access, but it does not own the run
/// lifecycle. This makes it the right type to move into async tasks, worker threads, or adapter
/// objects.
///
/// If the originating [`ExperimentRun`] is finished or dropped, existing handles become inactive
/// and will reject further operations.
#[derive(Clone)]
pub struct ExperimentRunHandle {
    metadata: ExperimentMetadata,
    inner: Weak<RunInner>,
    cancel_token: CancelToken,
}

struct RunInner {
    cancel_token: CancelToken,
    state: Mutex<RunState>,
    session: Box<dyn ExperimentSession>,
    reader: Box<dyn ExperimentArtifactReader>,
    activity_id_allocator: Arc<AtomicActivityIdAllocator>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunState {
    Active,
    Finished,
}

impl ExperimentRun {
    /// Create a run from backend-specific session and artifact reader implementations.
    ///
    /// This is the low-level constructor used by the built-in local and remote backends. Most
    /// callers should prefer [`Self::local`] or [`Self::remote`].
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
            cancel_token: cancel_token.clone(),
            state: Mutex::new(RunState::Active),
            session: Box::new(session),
            reader: Box::new(reader),
            activity_id_allocator: Arc::new(AtomicActivityIdAllocator::new()),
        });

        let handle = ExperimentRunHandle {
            metadata,
            inner: Arc::downgrade(&inner),
            cancel_token,
        };
        let tracing_registration = TracingRegistry::global().register_handle(handle.clone());

        Self {
            inner,
            handle,
            _tracing_registration: tracing_registration,
        }
    }

    /// Return a cancellation token that can be linked to child work.
    ///
    /// Cancelling the token does not finish the run; it only broadcasts cancellation to linked
    /// tasks and adapters.
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
    /// as successful by default. Any cloned [`ExperimentRunHandle`] becomes inactive afterwards.
    pub fn finish(self) -> Result<(), ExperimentError> {
        self.inner.finish_once(ExperimentCompletion::Success)
    }

    /// Mark the run as failed and finalize the backend session.
    ///
    /// Any cloned [`ExperimentRunHandle`] becomes inactive afterwards.
    pub fn fail(self, reason: impl Into<String>) -> Result<(), ExperimentError> {
        self.inner
            .finish_once(ExperimentCompletion::Failed(reason.into()))
    }
}

impl From<&ExperimentRun> for ExperimentRunHandle {
    fn from(value: &ExperimentRun) -> Self {
        value.handle()
    }
}

impl ExperimentRun {
    /// Borrow the identifier for the underlying run.
    pub fn id(&self) -> &ExperimentId {
        self.handle.id()
    }

    /// Log the serialized input arguments for the run.
    pub fn log_args<A: Serialize>(&self, args: &A) -> Result<(), ExperimentError> {
        self.handle.log_args(args)
    }

    /// Log a named configuration object for the run.
    pub fn log_config<C: Serialize>(
        &self,
        name: impl Into<String>,
        config: &C,
    ) -> Result<(), ExperimentError> {
        self.handle.log_config(name, config)
    }

    /// Log an informational message for the run.
    pub fn log_info(&self, message: impl Into<String>) -> Result<(), ExperimentError> {
        self.handle.log_info(message)
    }

    /// Log metric values for an epoch, split, and iteration.
    pub fn log_metric(
        &self,
        epoch: usize,
        split: impl Into<String>,
        iteration: usize,
        items: Vec<MetricValue>,
    ) -> Result<(), ExperimentError> {
        self.handle.log_metric(epoch, split, iteration, items)
    }

    /// Log a metric definition so later metric values have metadata attached.
    pub fn log_metric_definition(&self, spec: MetricSpec) -> Result<(), ExperimentError> {
        self.handle.log_metric_definition(spec)
    }

    /// Log aggregated metric values for an epoch and split.
    pub fn log_epoch_summary(
        &self,
        epoch: usize,
        split: impl Into<String>,
        items: Vec<MetricValue>,
    ) -> Result<(), ExperimentError> {
        self.handle.log_epoch_summary(epoch, split, items)
    }

    /// Encode and persist an artifact in the configured backend.
    pub fn save_artifact<E: BundleEncode>(
        &self,
        name: impl AsRef<str>,
        kind: ArtifactKind,
        artifact: E,
        settings: &E::Settings,
    ) -> Result<(), ExperimentError> {
        self.handle.save_artifact(name, kind, artifact, settings)
    }

    /// Load and decode an artifact from a compatible experiment identifier.
    pub fn use_artifact<D: BundleDecode>(
        &self,
        experiment_id: impl Into<ExperimentId>,
        name: impl AsRef<str>,
        settings: &D::Settings,
    ) -> Result<D, ExperimentError> {
        self.handle.use_artifact(experiment_id, name, settings)
    }

    /// Create an activity builder for the run with the provided name.
    ///
    /// Call [`ActivityBuilder::progress`] before starting when the activity should have a numeric
    /// meter.
    pub fn activity(&self, name: impl Into<String>) -> ActivityBuilder {
        self.handle.activity(name)
    }
}

/// Extension trait for cloning shareable handles from an [`ExperimentRun`].
///
/// Import this trait when you want a lightweight [`ExperimentRunHandle`] for async tasks, worker
/// threads, or adapter objects that should not own run finalization.
///
/// # Example
///
/// ```ignore
/// use tracel_experiment::{ExperimentRun, ExperimentRunHandleExt};
///
/// let run = ExperimentRun::local("./runs").unwrap();
/// let handle = run.handle();
///
/// std::thread::spawn(move || {
///     let _ = handle.log_info("worker started");
/// });
/// ```
pub trait ExperimentRunHandleExt {
    /// Clone a handle that can be shared across tasks and threads.
    fn handle(&self) -> ExperimentRunHandle;
}

impl ExperimentRunHandleExt for ExperimentRun {
    fn handle(&self) -> ExperimentRunHandle {
        self.handle.clone()
    }
}

impl ExperimentRunHandle {
    /// Borrow the identifier of the run this handle points to.
    pub fn id(&self) -> &ExperimentId {
        &self.metadata.id
    }

    /// Return a cancellation token that can be linked to child work.
    ///
    /// Cancelling the token does not finish the run; it only broadcasts cancellation to linked
    /// tasks and adapters.
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel_token.clone()
    }

    /// See [`ExperimentRun::log_args`].
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

    /// See [`ExperimentRun::log_config`].
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

    /// See [`ExperimentRun::log_info`].
    pub fn log_info(&self, message: impl Into<String>) -> Result<(), ExperimentError> {
        self.record_event(Event::Log {
            message: message.into(),
        })
    }

    /// See [`ExperimentRun::log_metric`].
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

    /// See [`ExperimentRun::log_metric_definition`].
    pub fn log_metric_definition(&self, spec: MetricSpec) -> Result<(), ExperimentError> {
        self.record_event(Event::MetricDefinition(spec))
    }

    /// See [`ExperimentRun::log_epoch_summary`].
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

    /// See [`ExperimentRun::save_artifact`].
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

    /// See [`ExperimentRun::use_artifact`].
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

    /// See [`ExperimentRun::activity`].
    ///
    /// If the originating run has already been finished or dropped, the activity builder will be a no-op.
    pub fn activity(&self, name: impl Into<String>) -> ActivityBuilder {
        let inner = match self.upgrade() {
            Ok(inner) => inner,
            Err(_) => {
                return ActivityBuilder::new(
                    Arc::new(|_| {}),
                    Arc::new(AtomicActivityIdAllocator::new()),
                    name.into(),
                );
            }
        };
        ActivityBuilder::new(
            Arc::new({
                let handle = self.clone();
                move |e| {
                    handle.record_event(Event::Activity(e)).ok();
                }
            }),
            inner.activity_id_allocator.clone(),
            name.into(),
        )
    }
}

impl ExperimentRunHandle {
    fn record_event(&self, event: Event) -> Result<(), ExperimentError> {
        let inner = self.upgrade()?;
        inner.ensure_active()?;
        inner.session.record_event(event)
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::activity::ActivityEvent;
    use crate::reader::{ExperimentReaderError, LoadedArtifact};
    use crate::session::BundleFn;

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
    fn run_forwards_handle_methods_for_event_recording() {
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

    #[test]
    fn run_metered_activity_start_records_progress_event() {
        let session = Arc::new(MockSession::default());
        let run = create_run(session.clone());

        let _progress = run.activity("load").progress().start();

        let events = session.events.lock().unwrap();
        assert!(matches!(
            events.as_slice(),
            [Event::Activity(ActivityEvent::Started { activity })] if activity.name == "load"
        ));
    }

    #[test]
    fn dropped_run_handle_activity_records_no_event() {
        let session = Arc::new(MockSession::default());
        let run = create_run(session.clone());
        let handle = run.handle();
        drop(run);

        let _progress = handle.activity("late").progress().start();

        let events = session.events.lock().unwrap();
        assert!(events.is_empty());
    }
}
