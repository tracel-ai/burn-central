use std::sync::Mutex;

use tracel_experiment::error::{ExperimentError, ExperimentErrorKind};
use tracel_experiment::session::{BundleFn, Event, ExperimentCompletion, ExperimentSession};
use tracel_experiment::{
    ActivityEvent, ActivityStatus, ArtifactKind, ExperimentRunControl, MetricSpec, MetricValue,
};

use crossbeam::channel::Sender;
use tracel_artifact::bundle::FsBundle;
use tracel_client::WebSocketClient;
use tracel_client::websocket::{
    ActivityEventRequest, ActivityMeterRequest, ActivityRequest, ActivityStatusRequest,
    ExperimentCompletion as RemoteExperimentCompletion, ExperimentMessage, InputUsed, MetricLog,
};

use super::log_store::LogUploader;
use super::log_store::TempLogStore;
use super::socket::ExperimentSocket;
use super::socket::ThreadError;

struct ActiveSession {
    sender: Sender<ExperimentMessage>,
    socket: ExperimentSocket,
}

#[derive(Debug, thiserror::Error)]
#[error("Failed to upload artifact: {message}")]
pub struct ArtifactUploadError {
    pub(crate) message: String,
    #[source]
    pub(crate) source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

pub trait ArtifactUploader {
    fn upload(
        &self,
        name: &str,
        kind: ArtifactKind,
        bundle: &FsBundle,
    ) -> Result<(), ArtifactUploadError>;
}

pub type BoxedArtifactUploader = Box<dyn ArtifactUploader + Send + Sync>;

pub struct RemoteExperimentSession {
    artifact_uploader: BoxedArtifactUploader,
    active: Mutex<Option<ActiveSession>>,
}

impl RemoteExperimentSession {
    pub fn new(
        log_uploader: Box<dyn LogUploader + Send>,
        artifact_uploader: Box<dyn ArtifactUploader + Send + Sync>,
        websocket: WebSocketClient,
        control: ExperimentRunControl,
    ) -> Self {
        let log_store = TempLogStore::new(log_uploader);
        let (sender, receiver) = crossbeam::channel::unbounded();
        let socket = ExperimentSocket::new(websocket, log_store, receiver, control);

        Self {
            artifact_uploader,
            active: Mutex::new(Some(ActiveSession { sender, socket })),
        }
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

impl ExperimentSession for RemoteExperimentSession {
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
            Event::Activity(activity_event) => {
                ExperimentMessage::Activity(to_remote_activity_event(activity_event))
            }
        };

        self.send(message)
    }

    fn save_artifact(
        &self,
        name: &str,
        kind: ArtifactKind,
        artifact: Box<BundleFn>,
    ) -> Result<(), ExperimentError> {
        let mut bundle = FsBundle::temp().map_err(|err| {
            ExperimentError::with_source(
                ExperimentErrorKind::Artifact,
                "Failed to create temporary bundle for artifact upload",
                err,
            )
        })?;

        artifact(&mut bundle)?;

        self.artifact_uploader
            .upload(name, kind, &bundle)
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

fn to_remote_metric_logs(items: Vec<MetricValue>) -> Vec<MetricLog> {
    items
        .into_iter()
        .map(|item| MetricLog {
            name: item.name,
            value: item.value,
        })
        .collect()
}

fn to_remote_activity_event(event: ActivityEvent) -> ActivityEventRequest {
    match event {
        ActivityEvent::Started { activity } => ActivityEventRequest::Started {
            activity: ActivityRequest {
                id: activity.id.as_u64(),
                parent: activity.parent.map(|parent| parent.as_u64()),
                name: activity.name,
                cancellable: activity.cancellable,
                meter: activity.meter.map(|meter| ActivityMeterRequest {
                    unit: meter.unit,
                    total: meter.total,
                }),
                attributes: activity.attributes,
            },
        },
        ActivityEvent::Updated { id, current } => ActivityEventRequest::Updated {
            id: id.as_u64(),
            current,
        },
        ActivityEvent::Message { id, message } => ActivityEventRequest::Message {
            id: id.as_u64(),
            message,
        },
        ActivityEvent::Finished {
            id,
            status,
            message,
        } => ActivityEventRequest::Finished {
            id: id.as_u64(),
            status: match status {
                ActivityStatus::Success => ActivityStatusRequest::Success,
                ActivityStatus::Abandoned => ActivityStatusRequest::Abandoned,
            },
            message,
        },
    }
}

fn to_remote_completion(completion: ExperimentCompletion) -> RemoteExperimentCompletion {
    match completion {
        ExperimentCompletion::Success => RemoteExperimentCompletion::Success,
        ExperimentCompletion::Failed(reason) => RemoteExperimentCompletion::Fail { reason },
        ExperimentCompletion::Cancelled => RemoteExperimentCompletion::Success,
    }
}
