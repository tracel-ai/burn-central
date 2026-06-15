use burn_central_client::request::{ArtifactFileSpecRequest, CreateArtifactRequest};
use burn_central_client::response::ArtifactResponse;
use burn_central_client::websocket::WebSocketError;
use burn_central_client::{Client, ClientError};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

use tracel_artifact::bundle::FsBundle;
use tracel_artifact::download::{ArtifactDownloadFile, DownloadError, download_artifacts_to_sink};
use tracel_artifact::upload::{
    MultipartUploadFile, MultipartUploadPart, UploadError, upload_bundle_multipart,
};

mod artifacts;
mod logs;

pub use artifacts::{ConsoleArtifactReader, ConsoleArtifactUploader};
pub use logs::ConsoleLogUploader;

use tracel_experiment::ArtifactKind;
use tracel_experiment::error::{ExperimentError, ExperimentErrorKind};
use tracel_experiment::{CancelToken, ExperimentId, ExperimentRun, ExperimentRunControl};

use tracel_experiment::ExperimentProvider;

use crate::backend::cloud::CloudBackend;
use crate::experiment::remote::session::RemoteExperimentSession;

#[derive(Debug, Clone)]
pub struct ExperimentPath {
    owner_name: String,
    project_name: String,
    experiment_num: i32,
}

impl ExperimentPath {
    pub fn new(
        owner_name: impl Into<String>,
        project_name: impl Into<String>,
        experiment_num: i32,
    ) -> Self {
        Self {
            owner_name: owner_name.into(),
            project_name: project_name.into(),
            experiment_num,
        }
    }

    pub fn owner_name(&self) -> &str {
        &self.owner_name
    }

    pub fn project_name(&self) -> &str {
        &self.project_name
    }

    pub fn experiment_num(&self) -> i32 {
        self.experiment_num
    }
}

/// A scope for artifact operations within a specific experiment.
#[derive(Clone)]
pub struct ExperimentArtifactClient {
    client: Client,
    exp_path: ExperimentPath,
}

impl ExperimentArtifactClient {
    pub fn new(client: Client, exp_path: ExperimentPath) -> Self {
        Self { client, exp_path }
    }

    pub fn upload(
        &self,
        name: impl Into<String>,
        kind: ArtifactKind,
        bundle: &FsBundle,
    ) -> Result<String, ArtifactError> {
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

        let res = self.client.create_artifact(
            self.exp_path.owner_name(),
            self.exp_path.project_name(),
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

        self.client.complete_artifact_upload(
            self.exp_path.owner_name(),
            self.exp_path.project_name(),
            self.exp_path.experiment_num(),
            &res.id,
            None,
        )?;

        Ok(res.id)
    }

    /// Download an artifact as a filesystem-backed bundle.
    pub fn download(&self, name: impl AsRef<str>) -> Result<FsBundle, ArtifactError> {
        let name = name.as_ref();
        let artifact = self.fetch(name)?;
        let resp = self.client.presign_artifact_download(
            self.exp_path.owner_name(),
            self.exp_path.project_name(),
            self.exp_path.experiment_num(),
            &artifact.id.to_string(),
        )?;

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
            .list_artifacts_by_name(
                self.exp_path.owner_name(),
                self.exp_path.project_name(),
                self.exp_path.experiment_num(),
                name,
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

#[derive(Debug, thiserror::Error)]
#[error(transparent)]
enum CloudError {
    Http(#[from] ClientError),
    WebSocket(#[from] WebSocketError),
}

impl ExperimentProvider for CloudBackend {
    fn create_experiment(
        &self,
        name: String,
        attributes: HashMap<String, Value>,
    ) -> Result<ExperimentRun, ExperimentError> {
        let digest = "46523358ec1646354ddab1cd8b93f2b920b44b24a26ea86c129d666d6bae2a5f".to_string();
        create_run(
            self.client.clone(),
            &self.namespace,
            &self.project,
            digest,
            name,
            attributes,
        )
        .map_err(|e| ExperimentError {
            kind: ExperimentErrorKind::Internal,
            message: "Failed to start Cloud experiment run".to_string(),
            source: Some(Box::new(e)),
        })
    }
}

pub fn create_cloud_experiment_run(
    client: Client,
    namespace: &str,
    project_name: &str,
    digest: String,
    routine: String,
) -> Result<ExperimentRun, Box<dyn std::error::Error + Send + Sync>> {
    Ok(create_run(
        client,
        namespace,
        project_name,
        digest,
        routine,
        HashMap::new(),
    )?)
}

fn create_run(
    client: Client,
    namespace: &str,
    project_name: &str,
    digest: String,
    routine: String,
    mut attributes: HashMap<String, Value>,
) -> Result<ExperimentRun, CloudError> {
    attributes
        .entry("code_version".to_string())
        .or_insert(Value::String(digest));
    attributes
        .entry("routine".to_string())
        .or_insert_with(|| Value::String(routine.clone()));

    let experiment =
        client.create_experiment(namespace, project_name, Some(routine), None, attributes)?;

    let experiment_num = experiment.experiment_num;
    let path = ExperimentPath::new(namespace, project_name, experiment_num);
    let cancel_token = CancelToken::new();
    let control = ExperimentRunControl::new(cancel_token.clone());

    let log_uploader = ConsoleLogUploader::new(client.clone(), path.clone());
    let artifact_uploader = ConsoleArtifactUploader::new(client.clone(), path.clone());

    let ws = client.create_experiment_run_websocket(namespace, project_name, experiment_num)?;

    let session = RemoteExperimentSession::new(
        Box::new(log_uploader),
        Box::new(artifact_uploader),
        ws,
        control.clone(),
    );

    let reader = ConsoleArtifactReader::new(client, path);
    let id = ExperimentId::from(format!("{}", experiment_num));

    Ok(ExperimentRun::new_with_control(
        id, session, reader, control,
    ))
}
