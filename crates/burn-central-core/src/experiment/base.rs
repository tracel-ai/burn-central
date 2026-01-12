use super::socket::ExperimentSocket;
use crate::artifacts::{ArtifactKind, ExperimentArtifactClient};
use crate::bundle::{BundleDecode, BundleEncode, InMemoryBundleReader};
use crate::experiment::error::ExperimentTrackerError;
use crate::experiment::log_store::TempLogStore;
use crate::experiment::socket::ThreadError;
use crate::schemas::ExperimentPath;
use burn_central_client::Client;
pub use burn_central_client::websocket::MetricLog;
use burn_central_client::websocket::{ExperimentCompletion, ExperimentMessage, InputUsed};
use crossbeam::channel::Sender;
use serde::Serialize;
use std::ops::Deref;
use std::sync::{Arc, Weak};

pub enum EndExperimentStatus {
    Success,
    Fail(String),
}

/// Represents a handle to an experiment, allowing logging of artifacts, metrics, and messages.
#[derive(Clone, Debug)]
pub struct ExperimentRunHandle {
    recorder: Weak<ExperimentRunInner>,
}

impl ExperimentRunHandle {
    fn try_upgrade(&self) -> Result<Arc<ExperimentRunInner>, ExperimentTrackerError> {
        self.recorder
            .upgrade()
            .ok_or(ExperimentTrackerError::InactiveExperiment)
    }

    /// Log arguments used to launch this experiment
    pub fn log_args<A: Serialize>(&self, args: &A) -> Result<(), ExperimentTrackerError> {
        self.try_upgrade()?.log_args(args)
    }

    /// Log an artifact with the given name, kind and settings.
    pub fn log_artifact<E: BundleEncode>(
        &self,
        name: impl Into<String>,
        kind: ArtifactKind,
        sources: E,
        settings: &E::Settings,
    ) -> Result<(), ExperimentTrackerError> {
        self.try_upgrade()?
            .log_artifact(name, kind, sources, settings)
    }

    /// Loads an artifact with the given name and settings.
    pub fn load_artifact<D: BundleDecode>(
        &self,
        name: impl AsRef<str>,
        settings: &D::Settings,
    ) -> Result<D, ExperimentTrackerError> {
        self.try_upgrade()?.load_artifact(name, settings)
    }

    /// Loads a raw artifact with the given name.
    pub fn load_artifact_raw(
        &self,
        name: impl AsRef<str>,
    ) -> Result<InMemoryBundleReader, ExperimentTrackerError> {
        self.try_upgrade()?.load_artifact_raw(name)
    }

    /// Logs a metric with the given name, epoch, iteration, value, and group.
    pub fn log_metric(
        &self,
        epoch: usize,
        split: impl Into<String>,
        iteration: usize,
        items: Vec<MetricLog>,
    ) {
        self.try_log_metric(epoch, split, iteration, items)
            .expect("Failed to log metric, experiment may have been closed or inactive");
    }

    /// Attempts to log a metric with the given name, epoch, iteration, value, and group.
    pub fn try_log_metric(
        &self,
        epoch: usize,
        split: impl Into<String>,
        iteration: usize,
        items: Vec<MetricLog>,
    ) -> Result<(), ExperimentTrackerError> {
        self.try_upgrade()?
            .log_metric(epoch, split, iteration, items)
    }

    pub fn log_metric_definition(
        &self,
        name: impl Into<String>,
        description: Option<String>,
        unit: Option<String>,
        higher_is_better: bool,
    ) -> Result<(), ExperimentTrackerError> {
        self.try_upgrade()?
            .log_metric_definition(name, description, unit, higher_is_better)
    }

    pub fn log_epoch_summary(
        &self,
        epoch: usize,
        split: String,
        best_metric_values: Vec<MetricLog>,
    ) -> Result<(), ExperimentTrackerError> {
        self.try_upgrade()?
            .log_epoch_summary(epoch, split, best_metric_values)
    }

    /// Logs an info message.
    pub fn log_info(&self, message: impl Into<String>) {
        self.try_log_info(message)
            .expect("Failed to log info, experiment may have been closed or inactive");
    }

    /// Attempts to log an info message.
    pub fn try_log_info(&self, message: impl Into<String>) -> Result<(), ExperimentTrackerError> {
        self.try_upgrade()?.log_info(message)
    }

    /// Logs an error message.
    pub fn log_error(&self, error: impl Into<String>) {
        self.try_log_error(error)
            .expect("Failed to log error, experiment may have been closed or inactive");
    }

    /// Attempts to log an error message.
    pub fn try_log_error(&self, error: impl Into<String>) -> Result<(), ExperimentTrackerError> {
        self.try_upgrade()?.log_error(error)
    }

    pub fn log_config<C: Serialize>(
        &self,
        name: impl Into<String>,
        config: &C,
    ) -> Result<(), ExperimentTrackerError> {
        self.try_upgrade()?.log_config(name.into(), config)
    }
}

/// Represents a recorder for an experiment, allowing logging of artifacts, metrics, and messages.
/// It is used internally by the [Experiment](ExperimentRun) struct to handle logging operations.
struct ExperimentRunInner {
    id: ExperimentPath,
    http_client: Client,
    sender: Sender<ExperimentMessage>,
}

impl ExperimentRunInner {
    fn send(&self, message: ExperimentMessage) -> Result<(), ExperimentTrackerError> {
        self.sender
            .send(message)
            .map_err(|_| ExperimentTrackerError::SocketClosed)
    }

    pub fn log_args<A: Serialize>(&self, args: &A) -> Result<(), ExperimentTrackerError> {
        let message = ExperimentMessage::Arguments(serde_json::to_value(args).map_err(|e| {
            ExperimentTrackerError::InternalError(format!("Failed to serialize arguments: {}", e))
        })?);
        self.send(message)
    }

    pub fn log_config<C: Serialize>(
        &self,
        name: String,
        config: &C,
    ) -> Result<(), ExperimentTrackerError> {
        let message = ExperimentMessage::Config {
            value: serde_json::to_value(config).map_err(|e| {
                ExperimentTrackerError::InternalError(format!("Failed to serialize config: {}", e))
            })?,
            name,
        };
        self.send(message)
    }

    pub fn log_artifact<E: BundleEncode>(
        &self,
        name: impl Into<String>,
        kind: ArtifactKind,
        artifact: E,
        settings: &E::Settings,
    ) -> Result<(), ExperimentTrackerError> {
        ExperimentArtifactClient::new(self.http_client.clone(), self.id.clone())
            .upload(name, kind, artifact, settings)
            .map_err(Into::into)
            .map(|_| ())
    }

    pub fn load_artifact_raw(
        &self,
        name: impl AsRef<str>,
    ) -> Result<InMemoryBundleReader, ExperimentTrackerError> {
        let scope = ExperimentArtifactClient::new(self.http_client.clone(), self.id.clone());
        let artifact = scope.fetch(&name)?;
        self.send(ExperimentMessage::InputUsed(InputUsed::Artifact {
            artifact_id: artifact.id.to_string(),
        }))?;
        scope.download_raw(name).map_err(Into::into)
    }

    pub fn load_artifact<D: BundleDecode>(
        &self,
        name: impl AsRef<str>,
        settings: &D::Settings,
    ) -> Result<D, ExperimentTrackerError> {
        let scope = ExperimentArtifactClient::new(self.http_client.clone(), self.id.clone());
        let artifact = scope.fetch(&name)?;
        self.send(ExperimentMessage::InputUsed(InputUsed::Artifact {
            artifact_id: artifact.id.to_string(),
        }))?;
        scope.download(name, settings).map_err(Into::into)
    }

    pub fn log_metric(
        &self,
        epoch: usize,
        split: impl Into<String>,
        iteration: usize,
        items: Vec<MetricLog>,
    ) -> Result<(), ExperimentTrackerError> {
        let message = ExperimentMessage::MetricsLog {
            epoch,
            split: split.into(),
            iteration,
            items,
        };
        self.send(message)
    }

    pub fn log_metric_definition(
        &self,
        name: impl Into<String>,
        description: Option<String>,
        unit: Option<String>,
        higher_is_better: bool,
    ) -> Result<(), ExperimentTrackerError> {
        let message = ExperimentMessage::MetricDefinitionLog {
            name: name.into(),
            description,
            unit,
            higher_is_better,
        };
        self.send(message)
    }

    pub fn log_epoch_summary(
        &self,
        epoch: usize,
        split: String,
        best_metric_values: Vec<MetricLog>,
    ) -> Result<(), ExperimentTrackerError> {
        let message = ExperimentMessage::EpochSummaryLog {
            epoch,
            split,
            best_metric_values,
        };
        self.send(message)
    }

    pub fn log_info(&self, message: impl Into<String>) -> Result<(), ExperimentTrackerError> {
        self.send(ExperimentMessage::Log(message.into()))
    }

    pub fn log_error(&self, error: impl Into<String>) -> Result<(), ExperimentTrackerError> {
        self.send(ExperimentMessage::Error(error.into()))
    }
}

/// Represents an experiment in Burn Central, which is a run of a machine learning model or process.
pub struct ExperimentRun {
    inner: Option<Arc<ExperimentRunInner>>,
    socket: Option<ExperimentSocket>,
    // temporary field to allow dereferencing to handle
    _handle: ExperimentRunHandle,
}

impl ExperimentRun {
    pub fn new(
        burn_client: Client,
        namespace: &str,
        project_name: &str,
        experiment_num: i32,
    ) -> Result<Self, ExperimentTrackerError> {
        let ws_client = burn_client
            .create_experiment_run_websocket(namespace, project_name, experiment_num)
            .map_err(|e| {
                ExperimentTrackerError::ConnectionFailed(format!(
                    "Failed to create WebSocket client: {}",
                    e
                ))
            })?;

        let experiment_path = ExperimentPath::new(namespace, project_name, experiment_num);

        let log_store = TempLogStore::new(burn_client.clone(), experiment_path.clone());
        let (sender, receiver) = crossbeam::channel::unbounded();
        let socket = ExperimentSocket::new(ws_client, log_store, receiver);

        let inner = Arc::new(ExperimentRunInner {
            id: experiment_path.clone(),
            http_client: burn_client.clone(),
            sender,
        });

        let _handle = ExperimentRunHandle {
            recorder: Arc::downgrade(&inner),
        };

        Ok(ExperimentRun {
            inner: Some(inner),
            socket: Some(socket),
            _handle,
        })
    }

    /// Returns a handle to the experiment, allowing logging of artifacts, metrics, and messages.
    pub fn handle(&self) -> ExperimentRunHandle {
        ExperimentRunHandle {
            recorder: Arc::downgrade(self.inner.as_ref().expect("Experiment already finished")),
        }
    }

    fn finish_internal(
        &mut self,
        end_status: EndExperimentStatus,
    ) -> Result<(), ExperimentTrackerError> {
        let socket = self
            .socket
            .take()
            .ok_or(ExperimentTrackerError::AlreadyFinished)?;

        let inner = self
            .inner
            .take()
            .ok_or(ExperimentTrackerError::AlreadyFinished)?;

        let completion = match end_status {
            EndExperimentStatus::Success => ExperimentCompletion::Success,
            EndExperimentStatus::Fail(reason) => ExperimentCompletion::Fail { reason },
        };

        inner
            .sender
            .send(ExperimentMessage::ExperimentComplete(completion))
            .map_err(|_| ExperimentTrackerError::SocketClosed)?;

        drop(inner);

        let thread_result = socket.join();

        match thread_result {
            Ok(_thread) => {}
            Err(ThreadError::WebSocket(msg)) => {
                eprintln!("Warning: WebSocket failure during experiment finish: {msg}");
            }
            Err(ThreadError::LogFlushError(msg)) => {
                eprintln!("Warning: Log artifact creation failed: {msg}");
            }
            Err(ThreadError::Panic) => {
                eprintln!("Warning: Experiment thread panicked");
                return Err(ExperimentTrackerError::InternalError(
                    "Experiment thread panicked".into(),
                ));
            }
        }

        Ok(())
    }

    pub fn finish(mut self) -> Result<(), ExperimentTrackerError> {
        self.finish_internal(EndExperimentStatus::Success)
    }

    pub fn fail(mut self, reason: impl Into<String>) -> Result<(), ExperimentTrackerError> {
        self.finish_internal(EndExperimentStatus::Fail(reason.into()))
    }
}

impl Drop for ExperimentRun {
    fn drop(&mut self) {
        if self.socket.is_some() {
            let _ = self.finish_internal(EndExperimentStatus::Fail(
                "Experiment dropped without finishing".to_string(),
            ));
        }
    }
}

/// Temporary implementation to allow dereferencing the Experiment to its recorder
/// This will be removed once the experiment logging api is completed
impl Deref for ExperimentRun {
    type Target = ExperimentRunHandle;

    fn deref(&self) -> &Self::Target {
        &self._handle
    }
}
