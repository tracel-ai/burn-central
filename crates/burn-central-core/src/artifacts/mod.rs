use burn_central_artifact::bundle::{BundleDecode, BundleEncode, FsBundle};
use burn_central_artifact::download::{
    ArtifactDownloadFile, DownloadError, download_artifacts_to_sink,
};
use burn_central_artifact::upload::{
    MultipartUploadFile, MultipartUploadPart, UploadError, upload_bundle_multipart,
};
use burn_central_client::request::{ArtifactFileSpecRequest, CreateArtifactRequest};
use burn_central_client::response::ArtifactResponse;
use burn_central_client::{Client, ClientError};
use std::collections::BTreeMap;

use crate::schemas::ExperimentPath;

#[derive(Debug, Clone, strum::Display, strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub enum ArtifactKind {
    Model,
    Log,
    Other,
}

/// A scope for artifact operations within a specific experiment.
#[derive(Clone)]
pub struct ExperimentArtifactClient {
    client: Client,
    exp_path: ExperimentPath,
}

impl ExperimentArtifactClient {
    pub(crate) fn new(client: Client, exp_path: ExperimentPath) -> Self {
        Self { client, exp_path }
    }

    /// Upload an artifact using a file-backed bundle sink to avoid loading files into memory.
    pub fn upload<E: BundleEncode>(
        &self,
        name: impl Into<String>,
        kind: ArtifactKind,
        artifact: E,
        settings: &E::Settings,
    ) -> Result<String, ArtifactError> {
        let name = name.into();
        let mut bundle = FsBundle::temp()
            .map_err(|e| ArtifactError::Internal(format!("Failed to create temp bundle: {e}")))?;

        artifact.encode(&mut bundle, settings).map_err(|e| {
            ArtifactError::Encoding(format!("Failed to encode artifact: {}", e.into()))
        })?;

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
                kind: kind.to_string(),
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
        upload_bundle_multipart(&bundle, &uploads)?;

        self.client.complete_artifact_upload(
            self.exp_path.owner_name(),
            self.exp_path.project_name(),
            self.exp_path.experiment_num(),
            &res.id,
            None,
        )?;

        Ok(res.id)
    }

    /// Download an artifact and decode it using the BundleDecode trait (filesystem-backed).
    pub fn download<D: BundleDecode>(
        &self,
        name: impl AsRef<str>,
        settings: &D::Settings,
    ) -> Result<D, ArtifactError> {
        let reader = self.download_raw(name.as_ref())?;
        D::decode(&reader, settings).map_err(|e| {
            ArtifactError::Decoding(format!(
                "Failed to decode artifact {}: {}",
                name.as_ref(),
                e.into()
            ))
        })
    }

    /// Download an artifact as a filesystem-backed bundle.
    pub fn download_raw(&self, name: impl AsRef<str>) -> Result<FsBundle, ArtifactError> {
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

#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("Artifact not found: {0}")]
    NotFound(String),
    #[error(transparent)]
    Client(#[from] ClientError),
    #[error("Error while encoding artifact: {0}")]
    Encoding(String),
    #[error("Error while decoding artifact: {0}")]
    Decoding(String),
    #[error(transparent)]
    Download(#[from] DownloadError),
    #[error(transparent)]
    Upload(#[from] UploadError),
    #[error("Internal error: {0}")]
    Internal(String),
}
