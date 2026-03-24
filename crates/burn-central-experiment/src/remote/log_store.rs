use burn_central_client::{
    Client, ClientError,
    request::{ArtifactFileSpecRequest, CreateArtifactRequest},
    response::MultipartUploadResponse,
};
use sha2::Digest;

use super::ExperimentPath;

#[derive(Debug)]
pub struct TempLogStore {
    logs: Vec<String>,
    client: Client,
    experiment_path: ExperimentPath,
    bytes: usize,
    artifact_id: Option<String>,
    file_counter: usize,
    num_digits: usize,
}

impl TempLogStore {
    // 10 MiB per chunk
    const CHUNK_SIZE: usize = 10 * 1024 * 1024;
    // Assume max 1000 log files (10GB of logs), use 3 digits padding
    const NUM_DIGITS: usize = 3;

    pub fn new(client: Client, experiment_path: ExperimentPath) -> TempLogStore {
        TempLogStore {
            logs: Vec::new(),
            client,
            experiment_path,
            bytes: 0,
            artifact_id: None,
            file_counter: 0,
            num_digits: Self::NUM_DIGITS,
        }
    }

    pub fn push(&mut self, log: String) -> Result<(), ClientError> {
        if self.bytes + log.len() > Self::CHUNK_SIZE {
            self.flush()?;
        }
        self.bytes += log.len();
        self.logs.push(log);
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), ClientError> {
        if self.logs.is_empty() {
            return Ok(());
        }

        let full_log = self.logs.join("");
        let log_bytes = full_log.as_bytes();
        let filename = format!(
            "experiment-{:0width$}.log",
            self.file_counter,
            width = self.num_digits
        );

        let checksum = format!("{:x}", sha2::Sha256::new_with_prefix(log_bytes).finalize());
        let file_spec = ArtifactFileSpecRequest {
            rel_path: filename.clone(),
            size_bytes: log_bytes.len() as u64,
            checksum,
        };

        let upload_url = if let Some(artifact_id) = &self.artifact_id {
            // Artifact exists, add files to it
            self.client
                .add_files_to_artifact(
                    self.experiment_path.owner_name(),
                    self.experiment_path.project_name(),
                    self.experiment_path.experiment_num(),
                    artifact_id,
                    vec![file_spec],
                )?
                .files
        } else {
            // First flush, create the artifact
            let response = self.client.create_artifact(
                self.experiment_path.owner_name(),
                self.experiment_path.project_name(),
                self.experiment_path.experiment_num(),
                CreateArtifactRequest {
                    name: "logs".to_string(),
                    kind: "log".to_string(),
                    files: vec![file_spec],
                },
            )?;

            // Store artifact ID for future flushes
            self.artifact_id = Some(response.id.clone());
            response.files
        };

        if let Some(file_response) = upload_url.first() {
            self.upload_chunk_multipart(log_bytes, &file_response.urls)?;
        } else {
            return Err(ClientError::UnknownError(
                "No upload URL returned for log file".to_string(),
            ));
        }

        if let Some(artifact_id) = &self.artifact_id {
            self.client.complete_artifact_upload(
                self.experiment_path.owner_name(),
                self.experiment_path.project_name(),
                self.experiment_path.experiment_num(),
                artifact_id,
                Some(vec![filename]),
            )?;
        }

        self.logs.clear();
        self.bytes = 0;
        self.file_counter += 1;

        Ok(())
    }

    fn upload_chunk_multipart(
        &self,
        chunk_data: &[u8],
        multipart_info: &MultipartUploadResponse,
    ) -> Result<(), ClientError> {
        let mut part_indices: Vec<usize> = (0..multipart_info.parts.len()).collect();
        part_indices.sort_by_key(|&i| multipart_info.parts[i].part);

        for (i, &part_idx) in part_indices.iter().enumerate() {
            let part = &multipart_info.parts[part_idx];
            if part.part != (i as u32 + 1) {
                return Err(ClientError::UnknownError(format!(
                    "Invalid part numbering: expected part {}, got part {}",
                    i + 1,
                    part.part
                )));
            }
        }

        let mut current_offset = 0usize;
        for &part_idx in part_indices.iter() {
            let part_info = &multipart_info.parts[part_idx];
            let end_offset = std::cmp::min(
                current_offset + part_info.size_bytes as usize,
                chunk_data.len(),
            );

            if current_offset >= chunk_data.len() {
                break;
            }

            let part_data = &chunk_data[current_offset..end_offset];
            self.client
                .upload_bytes_to_url(&part_info.url, part_data.to_vec())?;

            current_offset = end_offset;
        }

        Ok(())
    }
}
