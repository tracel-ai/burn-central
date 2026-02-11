use burn_central_client::ClientError;

/// Errors returned by the registry.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// Errors returned by the Burn Central HTTP client.
    #[error("Client error: {0}")]
    Client(#[from] ClientError),
    /// IO errors.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// HTTP download error.
    #[error("Download failed for {path}: {details}")]
    DownloadFailed { path: String, details: String },
    /// Manifest is missing or invalid.
    #[error("Invalid manifest: {0}")]
    InvalidManifest(String),
    /// Cache directory could not be resolved.
    #[error("Cache directory unavailable")]
    CacheDirUnavailable,
    /// Invalid path.
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    /// A file is missing from the cache.
    #[error("Missing file: {0}")]
    MissingFile(String),
    /// A file checksum does not match the manifest.
    #[error("Checksum mismatch for {path}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    /// A file size does not match the manifest.
    #[error("Size mismatch for {path}: expected {expected} bytes, got {actual} bytes")]
    SizeMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },
    /// Error while decoding a bundle from the cache.
    #[error("Decode error: {0}")]
    Decode(String),
}
