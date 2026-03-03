use std::io::{Read, Write};

use crossbeam::channel;
use reqwest::blocking::Client as HttpClient;
use sha2::Digest;

use crate::tools::validation::normalize_checksum;

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("failed to download {path}: {details}")]
    DownloadFailed { path: String, details: String },
    #[error("size mismatch for {path}: expected {expected} bytes, got {actual} bytes")]
    SizeMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },
    #[error("checksum mismatch for {path}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("invalid checksum: {0}")]
    InvalidChecksum(String),
    #[error("writer error: {0}")]
    WriterError(#[from] std::io::Error),
    #[error("invalid path: {0}")]
    InvalidPath(String),
}

/// A single file download task.
#[derive(Clone)]
pub struct DownloadTask<W> {
    pub rel_path: String,
    pub url: String,
    pub writer: W,
    pub expected_size: u64,
    pub expected_checksum: String,
}

/// Download multiple files in parallel.
pub fn download_tasks<W: Write + Send>(
    http: &HttpClient,
    tasks: Vec<DownloadTask<W>>,
    max_parallel: usize,
) -> Result<(), DownloadError> {
    if tasks.is_empty() {
        return Ok(());
    }

    if max_parallel <= 1 || tasks.len() == 1 {
        for mut task in tasks {
            download_one(http, &mut task)?;
        }
        return Ok(());
    }

    let (tx, rx) = channel::unbounded::<DownloadTask<W>>();
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
                for mut task in rx.iter() {
                    download_one(&http, &mut task)?;
                }
                Ok::<(), DownloadError>(())
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
fn download_one<W: Write>(
    http: &HttpClient,
    task: &mut DownloadTask<W>,
) -> Result<(), DownloadError> {
    let mut resp = http
        .get(&task.url)
        .send()
        .map_err(|e| DownloadError::DownloadFailed {
            path: task.rel_path.clone(),
            details: e.to_string(),
        })?;

    if !resp.status().is_success() {
        return Err(DownloadError::DownloadFailed {
            path: task.rel_path.clone(),
            details: format!("HTTP {}", resp.status()),
        });
    }

    let sink = &mut task.writer;
    let mut hasher = sha2::Sha256::new();
    let mut buf = [0u8; 1024 * 64];
    let mut total = 0u64;

    loop {
        let read = resp.read(&mut buf)?;
        if read == 0 {
            break;
        }
        sink.write_all(&buf[..read])?;
        hasher.update(&buf[..read]);
        total += read as u64;
    }

    let digest = format!("{:x}", hasher.finalize());
    let expected_checksum =
        normalize_checksum(&task.expected_checksum).map_err(DownloadError::InvalidChecksum)?;

    if total != task.expected_size {
        return Err(DownloadError::SizeMismatch {
            path: task.rel_path.clone(),
            expected: task.expected_size,
            actual: total,
        });
    }
    if digest != expected_checksum {
        return Err(DownloadError::ChecksumMismatch {
            path: task.rel_path.clone(),
            expected: expected_checksum,
            actual: digest,
        });
    }

    Ok(())
}
