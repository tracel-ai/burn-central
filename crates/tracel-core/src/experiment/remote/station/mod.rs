use std::collections::BTreeMap;
use tracel_artifact::bundle::FsBundle;
use tracel_artifact::download::{ArtifactDownloadFile, DownloadError, download_artifacts_to_sink};
use tracel_artifact::upload::{
    MultipartUploadFile, MultipartUploadPart, UploadError, upload_bundle_multipart,
};
use tracel_client::station::experiment::{
    ArtifactFileSpecRequest, ArtifactResponse, CompleteUploadRequest, CreateArtifactRequest,
    ListArtifactsQuery,
};
use tracel_client::websocket::WebSocketError;
use tracel_client::{ClientError, StationClient};

mod artifacts;
mod logs;

use artifacts::{StationArtifactReader, StationArtifactUploader};
use logs::StationLogUploader;

use std::collections::HashMap;

use serde_json::Value;
use tracel_client::station::experiment::CreateExperimentRequest;
use tracel_experiment::ArtifactKind;
use tracel_experiment::error::{ExperimentError, ExperimentErrorKind};
use tracel_experiment::{CancelToken, ExperimentId, ExperimentRun, ExperimentRunControl};

use tracel_experiment::ExperimentProvider;

use crate::backend::station::StationBackend;
use crate::experiment::remote::session::RemoteExperimentSession;

#[derive(Debug, thiserror::Error)]
enum StationError {
    #[error("Failed to create experiment on Station: check your Station URL and connectivity")]
    ExperimentCreation(#[from] ClientError),
    #[error("Failed to establish WebSocket connection to Station")]
    WebSocket(#[from] WebSocketError),
}

#[derive(Debug, Clone)]
pub struct ExperimentPath {
    experiment_num: i32,
}

impl ExperimentPath {
    pub fn new(experiment_num: i32) -> Self {
        Self { experiment_num }
    }

    pub fn experiment_num(&self) -> i32 {
        self.experiment_num
    }
}

/// A scope for artifact operations within a specific experiment.
#[derive(Clone)]
pub struct ExperimentArtifactClient {
    client: StationClient,
    exp_path: ExperimentPath,
}

impl ExperimentArtifactClient {
    pub fn new(client: StationClient, exp_path: ExperimentPath) -> Self {
        Self { client, exp_path }
    }

    pub fn upload(
        &self,
        name: impl Into<String>,
        kind: ArtifactKind,
        bundle: &FsBundle,
    ) -> Result<String, ArtifactError> {
        let client = self.client.experiments();

        let name = name.into();

        let mut specs = Vec::with_capacity(bundle.files().len());
        for f in bundle.files() {
            let size_bytes = f.size_bytes.ok_or_else(|| {
                ArtifactError::Internal(format!("Missing file size for {}", f.rel_path))
            })?;
            let checksum = f.checksum.clone().ok_or_else(|| {
                ArtifactError::Internal(format!("Missing checksum for {}", f.rel_path))
            })?;
            specs.push(ArtifactFileSpecRequest {
                rel_path: f.rel_path.clone(),
                size_bytes,
                checksum,
            });
        }

        let res = client.create_artifact(
            self.exp_path.experiment_num(),
            CreateArtifactRequest {
                name: name.clone(),
                kind: artifact_kind_name(kind).to_string(),
                files: specs,
            },
        )?;

        let mut multipart_map = BTreeMap::new();
        for f in &res.files {
            multipart_map.insert(f.rel_path.clone(), &f.urls);
        }

        let mut uploads = Vec::with_capacity(bundle.files().len());

        for f in bundle.files() {
            let multipart_info = multipart_map.get(&f.rel_path).ok_or_else(|| {
                ArtifactError::Internal(format!(
                    "Missing multipart upload info for file {}",
                    f.rel_path
                ))
            })?;

            let parts = multipart_info
                .parts
                .iter()
                .map(|part| MultipartUploadPart {
                    part: part.part,
                    url: part.url.clone(),
                    size_bytes: part.size_bytes,
                })
                .collect::<Vec<_>>();

            uploads.push(MultipartUploadFile {
                rel_path: f.rel_path.clone(),
                parts,
            });
        }
        upload_bundle_multipart(bundle, &uploads)?;

        client.complete_artifact_upload(
            self.exp_path.experiment_num(),
            &res.id,
            CompleteUploadRequest { file_names: None },
        )?;

        Ok(res.id)
    }

    /// Download an artifact as a filesystem-backed bundle.
    pub fn download(&self, name: impl AsRef<str>) -> Result<FsBundle, ArtifactError> {
        let name = name.as_ref();
        let artifact = self.fetch(name)?;
        let resp = self
            .client
            .experiments()
            .presign_artifact_download(self.exp_path.experiment_num(), artifact.id.to_string())?;

        let mut files = Vec::with_capacity(resp.files.len());
        for file in resp.files {
            files.push(ArtifactDownloadFile {
                rel_path: file.rel_path,
                url: file.url,
                size_bytes: None,
                checksum: None,
            });
        }

        let mut bundle = FsBundle::temp()
            .map_err(|e| ArtifactError::Internal(format!("Failed to create temp bundle: {e}")))?;

        download_artifacts_to_sink(&mut bundle, &files)?;

        Ok(bundle)
    }

    /// Fetch information about an artifact by name.
    pub fn fetch(&self, name: impl AsRef<str>) -> Result<ArtifactResponse, ArtifactError> {
        let name = name.as_ref();
        self.client
            .experiments()
            .list_artifacts(
                self.exp_path.experiment_num(),
                ListArtifactsQuery {
                    name: Some(name.to_string()),
                },
            )?
            .items
            .into_iter()
            .next()
            .ok_or_else(|| ArtifactError::NotFound(name.to_owned()))
    }
}

fn artifact_kind_name(kind: ArtifactKind) -> &'static str {
    match kind {
        ArtifactKind::Model => "model",
        ArtifactKind::Log => "log",
        ArtifactKind::Other => "other",
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("Artifact not found: {0}")]
    NotFound(String),
    #[error(transparent)]
    Client(#[from] ClientError),
    #[error(transparent)]
    Download(#[from] DownloadError),
    #[error(transparent)]
    Upload(#[from] UploadError),
    #[error("Internal error: {0}")]
    Internal(String),
}

impl ExperimentProvider for StationBackend {
    fn create_experiment(
        &self,
        name: String,
        attributes: HashMap<String, Value>,
    ) -> Result<ExperimentRun, ExperimentError> {
        create_run(self.client.clone(), name, attributes).map_err(|e| ExperimentError {
            kind: ExperimentErrorKind::Internal,
            message: "Failed to start Station experiment run".to_string(),
            source: Some(Box::new(e)),
        })
    }
}

fn create_run(
    client: StationClient,
    name: String,
    attributes: HashMap<String, Value>,
) -> Result<ExperimentRun, StationError> {
    let experiments_client = client.experiments();
    let experiment = experiments_client.create(CreateExperimentRequest {
        name: Some(name),
        description: None,
        attributes,
    })?;

    let experiment_num = experiment.experiment_num;
    let path = ExperimentPath::new(experiment_num);
    let cancel_token = CancelToken::new();
    let control = ExperimentRunControl::new(cancel_token.clone());

    let log_uploader = StationLogUploader::new(client.clone(), path.clone());
    let artifact_uploader = StationArtifactUploader::new(client.clone(), path);

    let ws = experiments_client.create_run_websocket(experiment_num)?;

    let session = RemoteExperimentSession::new(
        Box::new(log_uploader),
        Box::new(artifact_uploader),
        ws,
        control.clone(),
    );

    let reader = StationArtifactReader::new(client);
    let id = ExperimentId::from(format!("{}", experiment_num));

    Ok(ExperimentRun::new_with_control(
        id, session, reader, control,
    ))
}
