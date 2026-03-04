use std::fs;
use std::io;
use std::path::Path;

use burn_central_artifact::bundle::FsBundle;
use burn_central_artifact::download::{
    ArtifactDownloadFile, DownloadError, download_artifacts_to_sink,
};
use burn_central_artifact::normalize_checksum;
use burn_central_client::fleet::response::FleetModelDownloadResponse;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ModelCacheError {
    #[error("failed to access model cache filesystem: {0}")]
    Io(#[from] io::Error),
    #[error("failed to serialize model cache metadata: {0}")]
    Json(#[from] serde_json::Error),
    #[error("model version mismatch between sync ({expected}) and download ({actual})")]
    ModelVersionMismatch { expected: String, actual: String },
    #[error("no active model version in fleet state")]
    MissingActiveModelVersion,
    #[error("cached model file missing: {0}")]
    MissingCachedFile(String),
    #[error(transparent)]
    Download(#[from] DownloadError),
    #[error("invalid file path in model download manifest: {0}")]
    InvalidRelPath(String),
    #[error("invalid checksum in model metadata: {0}")]
    InvalidChecksum(String),
}

#[derive(Serialize, Deserialize)]
struct ModelDownloadManifest {
    model_version_id: String,
    files: Vec<ModelDownloadManifestFile>,
}

#[derive(Serialize, Deserialize)]
struct ModelDownloadManifestFile {
    rel_path: String,
    size_bytes: u64,
    checksum: String,
}

/// Ensure the model files from a download response are cached in the local filesystem, validating paths and content.
pub fn ensure_cached_model(
    models_root: &Path,
    expected_model_version_id: &str,
    download: &FleetModelDownloadResponse,
) -> Result<(), ModelCacheError> {
    if download.model_version_id != expected_model_version_id {
        tracing::error!(
            expected = expected_model_version_id,
            actual = download.model_version_id,
            "model version mismatch between sync and download"
        );
        return Err(ModelCacheError::ModelVersionMismatch {
            expected: expected_model_version_id.to_string(),
            actual: download.model_version_id.clone(),
        });
    }

    let model_root = models_root.join(&download.model_version_id);

    let manifest_path = model_root.join("manifest.json");
    if manifest_path.exists() {
        let bytes = fs::read(&manifest_path)?;
        let manifest: ModelDownloadManifest = serde_json::from_slice(&bytes)?;

        let mut manifest_files = manifest
            .files
            .iter()
            .map(|f| {
                Ok((
                    f.rel_path.clone(),
                    f.size_bytes,
                    normalize_checksum(&f.checksum).map_err(|e| {
                        ModelCacheError::InvalidChecksum(format!(
                            "manifest checksum for {}: {}",
                            f.rel_path, e
                        ))
                    })?,
                ))
            })
            .collect::<Result<Vec<_>, ModelCacheError>>()?;
        let mut download_files = download
            .files
            .iter()
            .map(|f| {
                Ok((
                    f.rel_path.clone(),
                    f.size_bytes,
                    normalize_checksum(&f.checksum).map_err(|e| {
                        ModelCacheError::InvalidChecksum(format!(
                            "download checksum for {}: {}",
                            f.rel_path, e
                        ))
                    })?,
                ))
            })
            .collect::<Result<Vec<_>, ModelCacheError>>()?;
        manifest_files.sort_unstable();
        download_files.sort_unstable();

        let manifest_matches_download = manifest.model_version_id == download.model_version_id
            && manifest_files == download_files;

        if manifest_matches_download
            && cached_files_present_and_sized(&model_root, &manifest.files)?
        {
            tracing::info!(
                version = %download.model_version_id,
                "cached model manifest matches download response and files are complete, skipping cache update"
            );
            return Ok(());
        }
    }

    let mut files = Vec::with_capacity(download.files.len());
    for entry in &download.files {
        files.push(ArtifactDownloadFile {
            rel_path: entry.rel_path.clone(),
            url: entry.url.clone(),
            size_bytes: Some(entry.size_bytes),
            checksum: Some(entry.checksum.clone()),
        });
    }

    tracing::info!(
        version = %download.model_version_id,
        num_files = files.len(),
        "new model version detected, downloading model files to local filesystem"
    );

    let mut sink = FsBundle::create(model_root.clone()).map_err(ModelCacheError::Io)?;
    download_artifacts_to_sink(&mut sink, &files)?;

    let manifest = ModelDownloadManifest {
        model_version_id: download.model_version_id.clone(),
        files: download
            .files
            .iter()
            .map(|f| ModelDownloadManifestFile {
                rel_path: f.rel_path.clone(),
                size_bytes: f.size_bytes,
                checksum: f.checksum.clone(),
            })
            .collect(),
    };

    write_manifest_if_changed(&manifest_path, &manifest)?;

    Ok(())
}

fn cached_files_present_and_sized(
    model_root: &Path,
    files: &[ModelDownloadManifestFile],
) -> Result<bool, io::Error> {
    for file in files {
        let path = model_root.join(&file.rel_path);
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(err) => return Err(err),
        };

        if !metadata.is_file() || metadata.len() != file.size_bytes {
            return Ok(false);
        }
    }

    Ok(true)
}

pub fn load_cached_model_source(
    models_root: &Path,
    model_version_id: &str,
) -> Result<FsBundle, ModelCacheError> {
    if model_version_id.is_empty() {
        tracing::error!("model version id is empty in fleet state");
        return Err(ModelCacheError::MissingActiveModelVersion);
    }

    tracing::info!(
        root = %models_root.display(),
        version = model_version_id,
        "reading model source metadata for model version"
    );

    let model_root = models_root.join(model_version_id);
    let manifest_path = model_root.join("manifest.json");

    if !manifest_path.exists() {
        tracing::error!(
            path = %manifest_path.display(),
            "cached model manifest not found for active model version"
        );
        return Err(ModelCacheError::MissingCachedFile(
            manifest_path.display().to_string(),
        ));
    }

    let bytes = fs::read(&manifest_path)?;
    let manifest: ModelDownloadManifest = serde_json::from_slice(&bytes)?;

    let files = manifest
        .files
        .iter()
        .map(|f| f.rel_path.clone())
        .collect::<Vec<_>>();

    for rel_path in &files {
        let file_path = model_root.join(rel_path);
        if !file_path.exists() {
            return Err(ModelCacheError::MissingCachedFile(
                file_path.display().to_string(),
            ));
        }
    }

    let source =
        FsBundle::with_files(model_root, files).map_err(ModelCacheError::InvalidRelPath)?;

    Ok(source)
}

fn write_manifest_if_changed(
    path: &Path,
    manifest: &ModelDownloadManifest,
) -> Result<bool, ModelCacheError> {
    let next = serde_json::to_vec_pretty(manifest)?;

    match fs::read(path) {
        Ok(current) if current == next => return Ok(false),
        Ok(_) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => return Err(ModelCacheError::Io(err)),
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = path.with_extension("tmp");
    fs::write(&tmp_path, &next)?;
    fs::rename(tmp_path, path)?;

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_central_client::fleet::response::FleetPresignedModelFileUrlResponse;
    use std::{
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "burn-central-runtime-{name}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    }

    #[test]
    fn creates_model_cache_layout() {
        let root = temp_path("model-cache");
        let download = FleetModelDownloadResponse {
            model_version_id: "mv-1".to_string(),
            files: vec![],
        };

        ensure_cached_model(&root, "mv-1", &download).expect("model should cache");
        let model_root = root.join("mv-1");
        assert!(model_root.exists());
        assert!(model_root.join("manifest.json").exists());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn cache_completeness_check_requires_files_with_expected_size() {
        let root = temp_path("cache-complete");
        let model_root = root.join("mv-1");
        fs::create_dir_all(&model_root).expect("model root should exist");

        let files = vec![ModelDownloadManifestFile {
            rel_path: "weights.bin".to_string(),
            size_bytes: 4,
            checksum: "abc".to_string(),
        }];

        assert!(
            !cached_files_present_and_sized(&model_root, &files).expect("check should succeed")
        );

        fs::write(model_root.join("weights.bin"), b"abc").expect("file should be created");
        assert!(
            !cached_files_present_and_sized(&model_root, &files).expect("check should succeed")
        );

        fs::write(model_root.join("weights.bin"), b"abcd").expect("file should be updated");
        assert!(cached_files_present_and_sized(&model_root, &files).expect("check should succeed"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn manifest_with_invalid_checksum_is_rejected() {
        let root = temp_path("invalid-checksum");
        let model_root = root.join("mv-1");
        fs::create_dir_all(&model_root).expect("model root should exist");

        let manifest = ModelDownloadManifest {
            model_version_id: "mv-1".to_string(),
            files: vec![ModelDownloadManifestFile {
                rel_path: "weights.bin".to_string(),
                size_bytes: 10,
                checksum: "md5:abc".to_string(),
            }],
        };
        let manifest_path = model_root.join("manifest.json");
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        let download = FleetModelDownloadResponse {
            model_version_id: "mv-1".to_string(),
            files: vec![FleetPresignedModelFileUrlResponse {
                rel_path: "weights.bin".to_string(),
                url: "mock://weights".to_string(),
                size_bytes: 10,
                checksum: "sha256:00".to_string(),
            }],
        };

        let err = ensure_cached_model(&root, "mv-1", &download)
            .expect_err("invalid checksum metadata should fail early");
        match err {
            ModelCacheError::InvalidChecksum(msg) => assert!(msg.contains("manifest checksum")),
            other => panic!("unexpected error: {other:?}"),
        }

        let _ = fs::remove_dir_all(root);
    }
}
