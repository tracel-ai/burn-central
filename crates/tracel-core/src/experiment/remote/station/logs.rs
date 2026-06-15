use std::collections::BTreeMap;

use sha2::Digest;
use tracel_artifact::{
    bundle::InMemoryBundleSources,
    upload::{MultipartUploadFile, MultipartUploadPart, upload_bundle_multipart},
};
use tracel_client::{
    ClientError, StationClient,
    station::experiment::{
        AddFilesRequest, CompleteUploadRequest,
        request::{ArtifactFileSpecRequest, CreateArtifactRequest},
    },
};

use crate::experiment::remote::log_store::{LogStoreError, LogUploader};

use super::ExperimentPath;

pub struct StationLogUploader {
    artifact_id: Option<String>,
    client: StationClient,
    experiment_path: ExperimentPath,
}

impl StationLogUploader {
    pub fn new(client: StationClient, experiment_path: ExperimentPath) -> Self {
        Self {
            artifact_id: None,
            client,
            experiment_path,
        }
    }
}

impl LogUploader for StationLogUploader {
    fn upload(&mut self, bundle: InMemoryBundleSources) -> Result<(), LogStoreError> {
        let client = self.client.experiments();
        let mut specs = Vec::with_capacity(bundle.files().len());
        for file in bundle.files() {
            let size_bytes = file.size();
            let checksum = format!(
                "{:x}",
                sha2::Sha256::new_with_prefix(file.source()).finalize()
            );
            let filename = file.dest_path().to_string();

            let file_spec = ArtifactFileSpecRequest {
                rel_path: filename.clone(),
                size_bytes: size_bytes as u64,
                checksum,
            };
            specs.push(file_spec);
        }

        let upload_urls = if let Some(artifact_id) = &self.artifact_id {
            // Artifact exists, add files to it
            client
                .add_artifact_files(
                    self.experiment_path.experiment_num(),
                    artifact_id,
                    AddFilesRequest { files: specs },
                )
                .map_err(|e| {
                    LogStoreError::new("Failed to add log files to artifact".to_string(), e)
                })?
                .files
        } else {
            // First flush, create the artifact
            let response = client
                .create_artifact(
                    self.experiment_path.experiment_num(),
                    CreateArtifactRequest {
                        name: "logs".to_string(),
                        kind: "log".to_string(),
                        files: specs,
                    },
                )
                .map_err(|e| LogStoreError::new("Failed to create log artifact".to_string(), e))?;

            // Store artifact ID for future flushes
            self.artifact_id = Some(response.id.clone());
            response.files
        };

        let mut multipart_map = BTreeMap::new();
        for file_response in &upload_urls {
            multipart_map.insert(file_response.rel_path.clone(), &file_response.urls);
        }

        let mut uploads = Vec::with_capacity(bundle.files().len());
        for file in bundle.files() {
            let filename = file.dest_path();
            let multipart_info = multipart_map
                .get(filename)
                .ok_or_else(|| {
                    ClientError::UnknownError(format!(
                        "Missing multipart upload info for file {}",
                        filename
                    ))
                })
                .map_err(|e| {
                    LogStoreError::new("Failed to get upload URLs for logs".to_string(), e)
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
                rel_path: filename.to_string(),
                parts,
            });
        }

        upload_bundle_multipart(&bundle, &uploads)
            .map_err(|e| LogStoreError::new("Failed to upload logs".to_string(), e))?;

        if let Some(artifact_id) = &self.artifact_id {
            client
                .complete_artifact_upload(
                    self.experiment_path.experiment_num(),
                    artifact_id,
                    CompleteUploadRequest {
                        file_names: Some(
                            bundle
                                .files()
                                .iter()
                                .map(|f| f.dest_path().to_string())
                                .collect(),
                        ),
                    },
                )
                .map_err(|e| {
                    LogStoreError::new("Failed to complete log artifact upload".to_string(), e)
                })?;
        }

        Ok(())
    }
}
