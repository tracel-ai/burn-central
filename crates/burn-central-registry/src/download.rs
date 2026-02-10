use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crossbeam::channel;
use reqwest::blocking::Client as HttpClient;
use sha2::Digest;

use crate::cache::normalize_checksum;
use crate::error::RegistryError;
use crate::manifest::ManifestFile;

/// A single file download task.
#[derive(Clone)]
pub struct DownloadTask {
    pub rel_path: String,
    pub url: String,
    pub dest: PathBuf,
    pub expected: ManifestFile,
}

/// Download multiple files in parallel.
pub fn download_tasks(
    http: &HttpClient,
    tasks: Vec<DownloadTask>,
    max_parallel: usize,
) -> Result<(), RegistryError> {
    if tasks.is_empty() {
        return Ok(());
    }

    if max_parallel <= 1 || tasks.len() == 1 {
        for task in tasks {
            download_one(http, &task)?;
        }
        return Ok(());
    }

    let (tx, rx) = channel::unbounded::<DownloadTask>();
    for task in tasks {
        tx.send(task).expect("channel open");
    }
    drop(tx);

    crossbeam::scope(|scope| {
        let mut handles = Vec::new();
        let worker_count = max_parallel.min(rx.len().max(1));
        for _ in 0..worker_count {
            let rx = rx.clone();
            let http = http.clone();
            handles.push(scope.spawn(move |_| {
                for task in rx.iter() {
                    download_one(&http, &task)?;
                }
                Ok::<(), RegistryError>(())
            }));
        }

        for handle in handles {
            handle.join().expect("thread panicked")?;
        }

        Ok(())
    })
    .expect("scope failed")
}

/// Download a single file with checksum verification.
fn download_one(http: &HttpClient, task: &DownloadTask) -> Result<(), RegistryError> {
    if let Some(parent) = task.dest.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp = temp_path(&task.dest)?;

    let mut resp = http
        .get(&task.url)
        .send()
        .map_err(|e| RegistryError::DownloadFailed {
            path: task.rel_path.clone(),
            details: e.to_string(),
        })?;

    if !resp.status().is_success() {
        return Err(RegistryError::DownloadFailed {
            path: task.rel_path.clone(),
            details: format!("HTTP {}", resp.status()),
        });
    }

    let mut file = File::create(&tmp)?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = [0u8; 1024 * 64];
    let mut total = 0u64;

    loop {
        let read = resp.read(&mut buf)?;
        if read == 0 {
            break;
        }
        file.write_all(&buf[..read])?;
        hasher.update(&buf[..read]);
        total += read as u64;
    }

    let digest = format!("{:x}", hasher.finalize());
    let expected_checksum = normalize_checksum(&task.expected.checksum)?;

    if total != task.expected.size_bytes {
        return Err(RegistryError::SizeMismatch {
            path: task.rel_path.clone(),
            expected: task.expected.size_bytes,
            actual: total,
        });
    }
    if digest != expected_checksum {
        return Err(RegistryError::ChecksumMismatch {
            path: task.rel_path.clone(),
            expected: expected_checksum,
            actual: digest,
        });
    }

    if task.dest.exists() {
        fs::remove_file(&task.dest)?;
    }

    fs::rename(tmp, &task.dest)?;

    Ok(())
}

/// Generate a temporary file path for downloads.
fn temp_path(dest: &Path) -> Result<PathBuf, RegistryError> {
    let file_name = dest
        .file_name()
        .ok_or_else(|| RegistryError::InvalidPath("missing file name".to_string()))?
        .to_string_lossy();
    Ok(dest.with_file_name(format!(".{file_name}.partial")))
}
