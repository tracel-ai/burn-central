use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use burn_central_core::bundle::normalize_bundle_path;

use crate::download::{DownloadError, DownloadTask, download_tasks};
use crate::tools::path::safe_join;

/// Generic download descriptor for any model artifact file.
#[derive(Debug, Clone)]
pub struct ArtifactDownloadFile {
    pub rel_path: String,
    pub url: String,
    pub size_bytes: u64,
    pub checksum: String,
}

/// Download artifact files into a destination directory, validating size and checksum.
pub fn download_artifacts_to_dir(
    dest_root: &Path,
    files: &[ArtifactDownloadFile],
) -> Result<(), DownloadError> {
    fs::create_dir_all(dest_root)?;

    if files.is_empty() {
        return Ok(());
    }

    let mut tmps = Vec::with_capacity(files.len());
    let mut tasks = Vec::with_capacity(files.len());
    for file in files {
        let rel_path = normalize_bundle_path(&file.rel_path);
        let dest = safe_join(dest_root, &rel_path)
            .map_err(|e| DownloadError::InvalidPath(e.to_string()))?;

        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = temp_path(&dest)?;
        tmps.push((dest.clone(), tmp.clone()));

        let dest_file = File::create(dest)?;
        let writer = BufWriter::new(dest_file);

        tasks.push(DownloadTask {
            rel_path: rel_path.clone(),
            url: file.url.clone(),
            writer,
            expected_size: file.size_bytes,
            expected_checksum: file.checksum.clone(),
        });
    }

    let parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let http = reqwest::blocking::Client::new();
    let res = download_tasks(&http, tasks, parallelism);

    for (tmp_dest, tmp) in tmps {
        if tmp_dest.exists() {
            fs::remove_file(&tmp_dest)?;
        }
        if tmp.exists() {
            fs::rename(tmp, tmp_dest)?;
        }
    }

    res
}

/// Generate a temporary file path for downloads.
fn temp_path(dest: &Path) -> Result<PathBuf, DownloadError> {
    let file_name = dest
        .file_name()
        .ok_or_else(|| DownloadError::InvalidPath("missing file name".to_string()))?
        .to_string_lossy();
    Ok(dest.with_file_name(format!(".{file_name}.partial")))
}
