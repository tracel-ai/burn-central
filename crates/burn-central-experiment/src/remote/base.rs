use std::sync::Mutex;

use crate::reader::{ArtifactRef, ExperimentArtifactReader, ExperimentReaderError, LoadedArtifact};
use crate::remote::RemoteExperimentId;
use crate::session::{Event, ExperimentCompletion, ExperimentSession};

use burn_central_artifact::bundle::FsBundle;
use burn_central_client::websocket::{
    ExperimentCompletion as RemoteExperimentCompletion, ExperimentMessage, InputUsed, MetricLog,
};
use burn_central_client::{Client, ClientError};
use crossbeam::channel::Sender;

use super::ExperimentPath;
use super::artifacts::ExperimentArtifactClient;
use super::log_store::TempLogStore;
use super::socket::ExperimentSocket;
use super::socket::ThreadError;
use crate::error::{ExperimentError, ExperimentErrorKind};
use crate::{ArtifactKind, CancelToken, ExperimentId, MetricSpec, MetricValue};

#[derive(Debug, thiserror::Error)]
pub enum BurnCentralError {
    /// Represents an error related to client operations.
    ///
    /// This error variant is used to encapsulate client-specific errors along with additional context
    /// and the underlying source error for more detailed debugging.
    ///
    /// # Fields
    /// - `context` (String): A description or additional information about the client error context.
    /// - `source` (ClientError): The underlying source of the client error, providing more details about the cause.
    #[error("Client error: {context}\nSource: {source}")]
    Client {
        context: String,
        source: ClientError,
    },
    /// Failed to connect the experiment run to the live backend stream.
    #[error("Failed to connect the experiment run to the server: {0}")]
    ExperimentConnection(String),
}

struct ActiveSession {
    sender: Sender<ExperimentMessage>,
    socket: ExperimentSocket,
}

pub struct BurnCentralSession {
    exp_path: ExperimentPath,
    http_client: Client,
    active: Mutex<Option<ActiveSession>>,
}

impl BurnCentralSession {
    pub fn new(
        burn_client: Client,
        experiment_path: ExperimentPath,
        cancel_token: CancelToken,
    ) -> Result<Self, BurnCentralError> {
        let ws_client = burn_client
            .create_experiment_run_websocket(
                experiment_path.owner_name(),
                experiment_path.project_name(),
                experiment_path.experiment_num(),
            )
            .map_err(|e| BurnCentralError::ExperimentConnection(e.to_string()))?;

        let log_store = TempLogStore::new(burn_client.clone(), experiment_path.clone());
        let (sender, receiver) = crossbeam::channel::unbounded();
        let socket = ExperimentSocket::new(ws_client, log_store, receiver, cancel_token);

        Ok(Self {
            exp_path: experiment_path,
            http_client: burn_client,
            active: Mutex::new(Some(ActiveSession { sender, socket })),
        })
    }

    fn send(&self, message: ExperimentMessage) -> Result<(), ExperimentError> {
        let guard = self.active.lock().unwrap();
        let active = guard.as_ref().ok_or_else(|| {
            ExperimentError::new(
                ExperimentErrorKind::AlreadyFinished,
                "Experiment run has already finished",
            )
        })?;

        active.sender.send(message).map_err(|_| {
            ExperimentError::new(
                ExperimentErrorKind::Internal,
                "Failed to send message to experiment session",
            )
        })
    }
}

impl ExperimentSession for BurnCentralSession {
    fn record_event(&self, event: Event) -> Result<(), ExperimentError> {
        let message = match event {
            Event::Args(value) => ExperimentMessage::Arguments(value),
            Event::Config { name, value } => ExperimentMessage::Config { name, value },
            Event::Log { message } => ExperimentMessage::Log(message),
            Event::Metrics {
                epoch,
                split,
                iteration,
                items,
            } => ExperimentMessage::MetricsLog {
                epoch,
                split,
                iteration,
                items: to_remote_metric_logs(items),
            },
            Event::MetricDefinition(MetricSpec {
                name,
                description,
                unit,
                higher_is_better,
            }) => ExperimentMessage::MetricDefinitionLog {
                name,
                description,
                unit,
                higher_is_better,
            },
            Event::EpochSummary {
                epoch,
                split,
                items,
            } => ExperimentMessage::EpochSummaryLog {
                epoch,
                split,
                best_metric_values: to_remote_metric_logs(items),
            },
            Event::ArtifactUsed {
                experiment_id: _,
                reference,
            } => ExperimentMessage::InputUsed(InputUsed::Artifact {
                artifact_id: reference.id,
            }),
        };

        self.send(message)
    }

    fn save_artifact(
        &self,
        name: &str,
        kind: ArtifactKind,
        artifact: &FsBundle,
    ) -> Result<(), ExperimentError> {
        ExperimentArtifactClient::new(self.http_client.clone(), self.exp_path.clone())
            .upload(name, kind, artifact)
            .map(|_| ())
            .map_err(|err| {
                ExperimentError::with_source(
                    ExperimentErrorKind::Artifact,
                    "Failed to upload experiment artifact",
                    err,
                )
            })
    }

    fn finish(&self, completion: ExperimentCompletion) -> Result<(), ExperimentError> {
        let active = self.active.lock().unwrap().take().ok_or_else(|| {
            ExperimentError::new(
                ExperimentErrorKind::AlreadyFinished,
                "Experiment run has already finished",
            )
        })?;

        let send_result =
            active
                .sender
                .send(ExperimentMessage::ExperimentComplete(to_remote_completion(
                    completion,
                )));
        drop(active.sender);

        let join_result = active.socket.join();

        if send_result.is_err() {
            return Err(ExperimentError::new(
                ExperimentErrorKind::Internal,
                "Failed to send experiment completion to remote session",
            ));
        }

        match join_result {
            Ok(_thread) => Ok(()),
            Err(ThreadError::WebSocket(err)) => {
                tracing::warn!("WebSocket failure during experiment finish: {err}");
                Ok(())
            }
            Err(ThreadError::LogFlushError(err)) => {
                tracing::warn!("Log artifact creation failed during experiment finish: {err}");
                Ok(())
            }
            Err(ThreadError::Panic) => Err(ExperimentError::new(
                ExperimentErrorKind::Internal,
                "Experiment background thread panicked",
            )),
        }
    }
}

pub struct BurnCentralArtifactReader {
    client: Client,
    exp_path: ExperimentPath,
}

impl BurnCentralArtifactReader {
    pub fn new(client: Client, exp_path: ExperimentPath) -> Self {
        Self { client, exp_path }
    }
}

impl ExperimentArtifactReader for BurnCentralArtifactReader {
    fn load_artifact_raw(
        &self,
        experiment_id: ExperimentId,
        name: &str,
    ) -> Result<LoadedArtifact, ExperimentReaderError> {
        let id = RemoteExperimentId::from_experiment_id(&experiment_id)
            .map_err(|_| ExperimentReaderError::new("Invalid experiment ID format"))?;

        let experiment_path = ExperimentPath::new(
            self.exp_path.owner_name().to_string(),
            self.exp_path.project_name().to_string(),
            id.num(),
        );
        let scope = ExperimentArtifactClient::new(self.client.clone(), experiment_path);
        let artifact = scope.fetch(name).map_err(|err| {
            ExperimentReaderError::with_source("Failed to resolve experiment artifact", err)
        })?;

        scope
            .download(name)
            .map_err(|err| {
                ExperimentReaderError::with_source("Failed to download experiment artifact", err)
            })
            .map(|bundle| {
                LoadedArtifact::new(
                    ArtifactRef {
                        id: artifact.id.to_string(),
                        name: name.to_string(),
                    },
                    bundle,
                )
            })
    }
}

fn to_remote_metric_logs(items: Vec<MetricValue>) -> Vec<MetricLog> {
    items
        .into_iter()
        .map(|item| MetricLog {
            name: item.name,
            value: item.value,
        })
        .collect()
}

fn to_remote_completion(completion: ExperimentCompletion) -> RemoteExperimentCompletion {
    match completion {
        ExperimentCompletion::Success => RemoteExperimentCompletion::Success,
        ExperimentCompletion::Failed(reason) => RemoteExperimentCompletion::Fail { reason },
        ExperimentCompletion::Cancelled => RemoteExperimentCompletion::Success,
    }
}
