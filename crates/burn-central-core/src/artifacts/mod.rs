use burn_central_client::response::{ArtifactResponse, MultipartUploadResponse};
use burn_central_client::{Client, ClientError};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::bundle::{BundleDecode, BundleEncode, FsBundleReader, FsBundleSink};
use crate::schemas::ExperimentPath;
use burn_central_client::request::{ArtifactFileSpecRequest, CreateArtifactRequest};

#[derive(Debug, Clone, strum::Display, strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub enum ArtifactKind {
    Model,
    Log,
    Other,
}

/// A scope for artifact operations within a specific experiment
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
        let mut sink = FsBundleSink::temp()
            .map_err(|e| ArtifactError::Internal(format!("Failed to create temp bundle: {e}")))?;

        artifact.encode(&mut sink, settings).map_err(|e| {
            ArtifactError::Encoding(format!("Failed to encode artifact: {}", e.into()))
        })?;

        let mut specs = Vec::with_capacity(sink.files().len());
        for f in sink.files() {
            specs.push(ArtifactFileSpecRequest {
                rel_path: f.rel_path.clone(),
                size_bytes: f.size_bytes,
                checksum: f.checksum.clone(),
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

        let mut multipart_map: BTreeMap<String, &MultipartUploadResponse> = BTreeMap::new();
        for f in &res.files {
            multipart_map.insert(f.rel_path.clone(), &f.urls);
        }

        let files = sink.files().to_vec();

        for f in files {
            let multipart_info = multipart_map.get(&f.rel_path).ok_or_else(|| {
                ArtifactError::Internal(format!(
                    "Missing multipart upload info for file {}",
                    f.rel_path
                ))
            })?;

            self.upload_file_multipart_streaming(&f.abs_path, &f.rel_path, multipart_info)?;
        }

        self.client.complete_artifact_upload(
            self.exp_path.owner_name(),
            self.exp_path.project_name(),
            self.exp_path.experiment_num(),
            &res.id,
            None,
        )?;

        Ok(res.id)
    }

    /// Download an artifact and decode it using the BundleDecode trait (filesystem-backed)
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

    /// Download an artifact as a filesystem-backed bundle reader
    pub fn download_raw(&self, name: impl AsRef<str>) -> Result<FsBundleReader, ArtifactError> {
        let name = name.as_ref();
        let artifact = self.fetch(name)?;
        let resp = self.client.presign_artifact_download(
            self.exp_path.owner_name(),
            self.exp_path.project_name(),
            self.exp_path.experiment_num(),
            &artifact.id.to_string(),
        )?;

        let mut file_list = Vec::new();
        for file_info in &resp.files {
            file_list.push(file_info.rel_path.clone());
        }

        // Create a temporary bundle reader that owns its temp directory
        let reader = FsBundleReader::temp(file_list)
            .map_err(|e| ArtifactError::Internal(format!("Failed to create temp bundle: {e}")))?;

        for file_info in resp.files {
            let rel_path = file_info.rel_path;
            let dest_path = reader.root().join(&rel_path);

            // Create parent directories
            if let Some(parent) = dest_path.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    ArtifactError::Internal(format!(
                        "Failed to create directory for {}: {}",
                        rel_path, e
                    ))
                })?;
            }

            // Download file directly to disk
            self.download_file_to_path(&file_info.url, &dest_path)?;
        }

        Ok(reader)
    }

    fn download_file_to_path(&self, url: &str, dest: &Path) -> Result<(), ArtifactError> {
        let http = reqwest::blocking::Client::new();
        let mut response = http
            .get(url)
            .send()
            .map_err(|e| ArtifactError::Internal(format!("Failed to download from URL: {}", e)))?;

        if !response.status().is_success() {
            return Err(ArtifactError::Internal(format!(
                "Failed to download file: HTTP {}",
                response.status()
            )));
        }

        let mut file = File::create(dest).map_err(|e| {
            ArtifactError::Internal(format!("Failed to create file {}: {}", dest.display(), e))
        })?;

        std::io::copy(&mut response, &mut file).map_err(|e| {
            ArtifactError::Internal(format!("Failed to write file {}: {}", dest.display(), e))
        })?;

        Ok(())
    }

    /// Fetch information about an artifact by name
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

    fn upload_file_multipart_streaming(
        &self,
        file_path: &Path,
        rel_path: &str,
        multipart_info: &MultipartUploadResponse,
    ) -> Result<(), ArtifactError> {
        let metadata = fs::metadata(file_path)
            .map_err(|e| ArtifactError::Internal(format!("Failed to stat file {rel_path}: {e}")))?;
        let file_len = metadata.len();

        let mut part_indices: Vec<usize> = (0..multipart_info.parts.len()).collect();
        part_indices.sort_by_key(|&i| multipart_info.parts[i].part);

        for (i, &part_idx) in part_indices.iter().enumerate() {
            let part = &multipart_info.parts[part_idx];
            if part.part != (i as u32 + 1) {
                return Err(ArtifactError::Internal(format!(
                    "Invalid part numbering for {}: expected part {}, got part {}",
                    rel_path,
                    i + 1,
                    part.part
                )));
            }
        }

        let http = reqwest::blocking::Client::new();
        let mut offset = 0u64;

        for (part_index, &part_idx) in part_indices.iter().enumerate() {
            let part_info = &multipart_info.parts[part_idx];
            let size = part_info.size_bytes;

            if offset + size > file_len {
                return Err(ArtifactError::Internal(format!(
                    "Part {} exceeds file length for {}",
                    part_index + 1,
                    rel_path
                )));
            }

            let mut file = File::open(file_path).map_err(|e| {
                ArtifactError::Internal(format!("Failed to open file {rel_path}: {e}"))
            })?;
            file.seek(SeekFrom::Start(offset)).map_err(|e| {
                ArtifactError::Internal(format!("Failed to seek file {rel_path}: {e}"))
            })?;

            let reader = file.take(size);
            let body = reqwest::blocking::Body::sized(reader, size);
            let response = http.put(&part_info.url).body(body).send().map_err(|e| {
                ArtifactError::Internal(format!(
                    "Failed to upload part {} of {} for {}: {}",
                    part_index + 1,
                    multipart_info.parts.len(),
                    rel_path,
                    e
                ))
            })?;

            if !response.status().is_success() {
                return Err(ArtifactError::Internal(format!(
                    "Failed to upload part {} of {} for {}: HTTP {}",
                    part_index + 1,
                    multipart_info.parts.len(),
                    rel_path,
                    response.status()
                )));
            }

            offset += size;
        }

        if offset != file_len {
            return Err(ArtifactError::Internal(format!(
                "Multipart upload size mismatch for {} (uploaded {}, expected {})",
                rel_path, offset, file_len
            )));
        }

        Ok(())
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
    #[error("Internal error: {0}")]
    Internal(String),
}
