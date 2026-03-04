//! This module provides utilities for downloading artifact files from any source to any target bundle sink.
//!
//! Downloaded files are validated against expected sizes and checksums when provided, and the download process can be customized with any implementation of the FileTransferBackend trait (e.g. for custom HTTP clients, authentication, retries, etc).

use std::collections::HashSet;
use std::io::Read;

use sha2::Digest;

use crate::bundle::BundleSink;
use crate::tools::path::normalize_bundle_path;
use crate::tools::validation::normalize_checksum;
use crate::{FileTransferBackend, ReqwestTransferBackend};

/// Errors that can occur during artifact file downloads.
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    /// Errors from the transfer backend (e.g. network errors, HTTP errors).
    #[error("transfer error for {rel_path}: {source}")]
    Transfer {
        rel_path: String,
        #[source]
        source: crate::transfer::TransferError,
    },
    /// Errors related to file size mismatches after download.
    #[error("size mismatch for {path}: expected {expected} bytes, got {actual} bytes")]
    SizeMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },
    /// Errors related to checksum mismatches after download.
    #[error("checksum mismatch for {path}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    /// Errors related to invalid checksums (e.g. non-hex, wrong length).
    #[error("invalid checksum: {0}")]
    InvalidChecksum(String),
    /// Errors related to invalid relative paths (e.g. empty, duplicates, unsafe).
    #[error("invalid path: {0}")]
    InvalidPath(String),
    /// Errors from the target bundle sink (e.g. file system errors).
    #[error("target error: {0}")]
    TargetError(String),
}

/// Generic download descriptor for any model artifact file.
#[derive(Debug, Clone)]
pub struct ArtifactDownloadFile {
    pub rel_path: String,
    pub url: String,
    /// Optional expected file size in bytes.
    pub size_bytes: Option<u64>,
    /// Optional expected SHA-256 checksum.
    pub checksum: Option<String>,
}

/// Download artifact files into any bundle sink implementation.
pub fn download_artifacts_to_sink<S: BundleSink>(
    sink: &mut S,
    files: &[ArtifactDownloadFile],
) -> Result<(), DownloadError> {
    let backend = ReqwestTransferBackend::new();
    download_artifacts_to_sink_with_backend(&backend, sink, files)
}

/// Download artifact files into any bundle sink implementation using a custom transfer backend.
pub fn download_artifacts_to_sink_with_backend<B: FileTransferBackend, S: BundleSink>(
    backend: &B,
    sink: &mut S,
    files: &[ArtifactDownloadFile],
) -> Result<(), DownloadError> {
    let files = validated_download_files(files)?;
    for (rel_path, file) in files {
        let reader = backend
            .get_reader(&file.url)
            .map_err(|e| DownloadError::Transfer {
                rel_path: rel_path.clone(),
                source: e,
            })?;
        let mut verifying_reader = VerifyingReader::new(reader);

        sink.put_file(&rel_path, &mut verifying_reader)
            .map_err(DownloadError::TargetError)?;

        let (total, digest) = verifying_reader.finish();
        validate_download(
            &rel_path,
            total,
            digest,
            file.size_bytes,
            file.checksum.as_deref(),
        )?;
    }

    Ok(())
}

fn validated_download_files(
    files: &[ArtifactDownloadFile],
) -> Result<Vec<(String, &ArtifactDownloadFile)>, DownloadError> {
    let mut seen = HashSet::with_capacity(files.len());
    let mut out = Vec::with_capacity(files.len());
    for file in files {
        let rel_path = normalize_bundle_path(&file.rel_path);
        if rel_path.is_empty() {
            return Err(DownloadError::InvalidPath(
                "empty relative artifact path".to_string(),
            ));
        }
        if !seen.insert(rel_path.clone()) {
            return Err(DownloadError::InvalidPath(format!(
                "duplicate relative artifact path: {rel_path}"
            )));
        }

        out.push((rel_path, file));
    }

    Ok(out)
}

struct VerifyingReader<R: Read> {
    inner: R,
    hasher: sha2::Sha256,
    total: u64,
}

impl<R: Read> VerifyingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: sha2::Sha256::new(),
            total: 0,
        }
    }

    fn finish(self) -> (u64, String) {
        (self.total, format!("{:x}", self.hasher.finalize()))
    }
}

impl<R: Read> Read for VerifyingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let read = self.inner.read(buf)?;
        self.hasher.update(&buf[..read]);
        self.total += read as u64;
        Ok(read)
    }
}

fn validate_download(
    rel_path: &str,
    total: u64,
    digest: String,
    expected_size: Option<u64>,
    expected_checksum: Option<&str>,
) -> Result<(), DownloadError> {
    if let Some(expected_size) = expected_size {
        if total != expected_size {
            return Err(DownloadError::SizeMismatch {
                path: rel_path.to_string(),
                expected: expected_size,
                actual: total,
            });
        }
    }

    if let Some(expected_checksum) = expected_checksum {
        let expected_checksum =
            normalize_checksum(expected_checksum).map_err(DownloadError::InvalidChecksum)?;
        if digest != expected_checksum {
            return Err(DownloadError::ChecksumMismatch {
                path: rel_path.to_string(),
                expected: expected_checksum,
                actual: digest,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::InMemoryBundleSources;
    use crate::transfer::TransferError;
    use std::collections::HashMap;
    use std::io::{Cursor, Read};
    use std::sync::Arc;

    #[derive(Clone)]
    struct MockBackend {
        files: Arc<HashMap<String, Vec<u8>>>,
    }

    impl MockBackend {
        fn new(files: HashMap<String, Vec<u8>>) -> Self {
            Self {
                files: Arc::new(files),
            }
        }
    }

    impl FileTransferBackend for MockBackend {
        fn put_reader<R: Read + Send + 'static>(
            &self,
            _url: &str,
            mut reader: R,
            _size_bytes: u64,
        ) -> Result<(), TransferError> {
            let mut buf = Vec::new();
            reader
                .read_to_end(&mut buf)
                .map_err(|e| TransferError::Transport(e.to_string()))?;
            Ok(())
        }

        fn get_reader(&self, url: &str) -> Result<Box<dyn Read + Send>, TransferError> {
            let bytes = self
                .files
                .get(url)
                .ok_or_else(|| TransferError::Transport(format!("missing url in mock: {url}")))?;
            Ok(Box::new(Cursor::new(bytes.clone())))
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = sha2::Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    #[test]
    fn downloads_to_sink_and_validates_checksum_and_size() {
        let data = b"hello world".to_vec();
        let checksum = sha256_hex(&data);
        let mut sink = InMemoryBundleSources::new();
        let backend = MockBackend::new(HashMap::from([("mock://f1".to_string(), data.clone())]));
        let files = vec![ArtifactDownloadFile {
            rel_path: "weights.bin".to_string(),
            url: "mock://f1".to_string(),
            size_bytes: Some(data.len() as u64),
            checksum: Some(checksum),
        }];

        download_artifacts_to_sink_with_backend(&backend, &mut sink, &files)
            .expect("download should succeed");

        assert_eq!(sink.len(), 1);
        assert_eq!(sink.files()[0].dest_path(), "weights.bin");
        assert_eq!(sink.files()[0].source(), data);
    }

    #[test]
    fn rejects_duplicate_relative_paths() {
        let backend = MockBackend::new(HashMap::new());
        let mut sink = InMemoryBundleSources::new();
        let files = vec![
            ArtifactDownloadFile {
                rel_path: "a.bin".to_string(),
                url: "mock://a".to_string(),
                size_bytes: None,
                checksum: None,
            },
            ArtifactDownloadFile {
                rel_path: "a.bin".to_string(),
                url: "mock://b".to_string(),
                size_bytes: None,
                checksum: None,
            },
        ];

        let err = download_artifacts_to_sink_with_backend(&backend, &mut sink, &files)
            .expect_err("duplicate paths should fail");

        match err {
            DownloadError::InvalidPath(msg) => assert!(msg.contains("duplicate")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn fails_on_checksum_mismatch() {
        let data = b"payload".to_vec();
        let mut sink = InMemoryBundleSources::new();
        let backend = MockBackend::new(HashMap::from([("mock://f2".to_string(), data.clone())]));
        let files = vec![ArtifactDownloadFile {
            rel_path: "params.bin".to_string(),
            url: "mock://f2".to_string(),
            size_bytes: Some(data.len() as u64),
            checksum: Some("00".repeat(32)),
        }];

        let err = download_artifacts_to_sink_with_backend(&backend, &mut sink, &files)
            .expect_err("checksum mismatch should fail");

        match err {
            DownloadError::ChecksumMismatch { path, .. } => assert_eq!(path, "params.bin"),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
