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

    let mut staged_files = Vec::with_capacity(files.len());
    let mut tasks = Vec::with_capacity(files.len());
    for file in files {
        let rel_path = normalize_bundle_path(&file.rel_path);
        let dest = safe_join(dest_root, &rel_path)
            .map_err(|e| DownloadError::InvalidPath(e.to_string()))?;

        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = temp_path(&dest)?;
        if tmp.exists() {
            fs::remove_file(&tmp)?;
        }
        staged_files.push((dest.clone(), tmp.clone(), rel_path.clone()));

        let tmp_file = File::create(tmp)?;
        let writer = BufWriter::new(tmp_file);

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
    if let Err(err) = download_tasks(&http, tasks, parallelism) {
        for (_, tmp, _) in staged_files {
            let _ = fs::remove_file(tmp);
        }
        return Err(err);
    }

    for (dest, tmp, rel_path) in staged_files {
        if !tmp.exists() {
            return Err(DownloadError::DownloadFailed {
                path: rel_path,
                details: "temporary downloaded file is missing".to_string(),
            });
        }
        if dest.exists() {
            fs::remove_file(&dest)?;
        }
        fs::rename(tmp, dest)?;
    }

    Ok(())
}

/// Generate a temporary file path for downloads.
fn temp_path(dest: &Path) -> Result<PathBuf, DownloadError> {
    let file_name = dest
        .file_name()
        .ok_or_else(|| DownloadError::InvalidPath("missing file name".to_string()))?
        .to_string_lossy();
    Ok(dest.with_file_name(format!(".{file_name}.partial")))
}
