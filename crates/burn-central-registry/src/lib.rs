//! Local registry/cache for Burn Central model artifacts.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use burn_central_client::{BurnCentralCredentials, Client, ClientError, Env};
use burn_central_core::bundle::{normalize_bundle_path, BundleDecode, BundleSource};
use crossbeam::channel;
use directories::{BaseDirs, ProjectDirs};
use reqwest::blocking::Client as HttpClient;
use url::Url;
use serde::{Deserialize, Serialize};
use sha2::Digest;

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

/// Configuration for the registry cache.
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Root directory for the cache.
    pub cache_dir: PathBuf,
    /// Max number of parallel downloads.
    pub max_parallel: usize,
    /// Whether to verify file checksums on cache hit.
    pub verify_checksums: bool,
}

impl RegistryConfig {
    /// Build a default configuration using the platform cache directory.
    pub fn new() -> Result<Self, RegistryError> {
        let cache_dir = default_cache_dir()?;
        Ok(Self {
            cache_dir,
            max_parallel: default_parallelism(),
            verify_checksums: true,
        })
    }

    /// Override the cache directory.
    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = dir.into();
        self
    }

    /// Override the max number of parallel downloads.
    pub fn with_max_parallel(mut self, max_parallel: usize) -> Self {
        self.max_parallel = max_parallel.max(1);
        self
    }

    /// Toggle checksum verification on cache hits.
    pub fn with_verify_checksums(mut self, verify: bool) -> Self {
        self.verify_checksums = verify;
        self
    }
}

fn default_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn default_cache_dir() -> Result<PathBuf, RegistryError> {
    if let Some(project) = ProjectDirs::from("ai", "tracel", "burn-central") {
        return Ok(project.cache_dir().join("registry"));
    }
    if let Some(base) = BaseDirs::new() {
        return Ok(base.cache_dir().join("burn-central").join("registry"));
    }
    Err(RegistryError::CacheDirUnavailable)
}

/// Builder for a registry client.
#[derive(Debug, Clone)]
pub struct RegistryBuilder {
    credentials: BurnCentralCredentials,
    env: Option<Env>,
    endpoint: Option<Url>,
    config: Option<RegistryConfig>,
}

impl RegistryBuilder {
    /// Create a new registry builder with the given credentials.
    pub fn new(credentials: impl Into<BurnCentralCredentials>) -> Self {
        Self {
            credentials: credentials.into(),
            env: None,
            endpoint: None,
            config: None,
        }
    }

    /// Use a specific environment (production by default).
    pub fn with_env(mut self, env: Env) -> Self {
        self.env = Some(env);
        self
    }

    /// Use a specific endpoint URL (self-hosted).
    pub fn with_endpoint(mut self, endpoint: Url) -> Self {
        self.endpoint = Some(endpoint);
        self
    }

    /// Use the provided registry configuration.
    pub fn with_config(mut self, config: RegistryConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the registry client.
    pub fn build(self) -> Result<Registry, RegistryError> {
        let config = match self.config {
            Some(cfg) => cfg,
            None => RegistryConfig::new()?,
        };

        let client = if let Some(url) = self.endpoint {
            #[allow(deprecated)]
            Client::from_url(url, &self.credentials)?
        } else {
            let env = self.env.unwrap_or(Env::Production);
            Client::new(env, &self.credentials)?
        };

        Registry::from_client(client, config)
    }
}

/// Registry client for downloading and caching model artifacts.
#[derive(Clone)]
pub struct Registry {
    client: Client,
    http: HttpClient,
    config: RegistryConfig,
}

impl Registry {
    /// Create a registry client from a Burn Central HTTP client and config.
    pub fn from_client(client: Client, config: RegistryConfig) -> Result<Self, RegistryError> {
        fs::create_dir_all(&config.cache_dir)?;
        Ok(Self {
            client,
            http: HttpClient::new(),
            config,
        })
    }

    /// Create a model handle scoped to a project.
    pub fn model(
        &self,
        namespace: impl Into<String>,
        project: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<ModelHandle, RegistryError> {
        let model = ModelRef::new(namespace, project, model)?;
        Ok(ModelHandle {
            registry: self.clone(),
            model,
        })
    }
}

/// Reference to a model in the registry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelRef {
    namespace: String,
    project: String,
    model: String,
}

impl ModelRef {
    fn new(
        namespace: impl Into<String>,
        project: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self, RegistryError> {
        let namespace = namespace.into();
        let project = project.into();
        let model = model.into();
        validate_path_component(&namespace, "namespace")?;
        validate_path_component(&project, "project")?;
        validate_path_component(&model, "model")?;
        Ok(Self {
            namespace,
            project,
            model,
        })
    }

    fn version_dir(&self, root: &Path, version: u32) -> PathBuf {
        root.join(&self.namespace)
            .join(&self.project)
            .join("models")
            .join(&self.model)
            .join("versions")
            .join(version.to_string())
    }
}

fn validate_path_component(value: &str, label: &str) -> Result<(), RegistryError> {
    if value.is_empty() {
        return Err(RegistryError::InvalidPath(format!(
            "{label} must not be empty"
        )));
    }
    let path = Path::new(value);
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(RegistryError::InvalidPath(format!(
                    "{label} contains invalid path segments"
                )))
            }
            Component::Normal(_) => {}
            Component::CurDir => {
                return Err(RegistryError::InvalidPath(format!(
                    "{label} contains invalid path segments"
                )))
            }
        }
    }
    Ok(())
}

/// Handle for downloading and caching a specific model.
#[derive(Clone)]
pub struct ModelHandle {
    registry: Registry,
    model: ModelRef,
}

impl ModelHandle {
    /// Ensure a model version is cached locally.
    pub fn ensure(&self, version: u32) -> Result<CachedModel, RegistryError> {
        let version_dir = self.model.version_dir(&self.registry.config.cache_dir, version);

        if let Ok(manifest) = load_manifest(&version_dir) {
            if cache_is_valid(
                &version_dir,
                &manifest,
                self.registry.config.verify_checksums,
            )? {
                return Ok(CachedModel::new(version_dir, manifest));
            }
        }

        self.download_version(version, &version_dir)
    }

    /// Download and decode a model version using the bundle decoder.
    pub fn decode<T: BundleDecode>(
        &self,
        version: u32,
        settings: &T::Settings,
    ) -> Result<T, RegistryError> {
        let cached = self.ensure(version)?;
        T::decode(&cached.reader(), settings)
            .map_err(|e| RegistryError::Decode(e.into().to_string()))
    }

    fn download_version(
        &self,
        version: u32,
        version_dir: &Path,
    ) -> Result<CachedModel, RegistryError> {
        fs::create_dir_all(version_dir)?;

        let version_info = self.registry.client.get_model_version(
            &self.model.namespace,
            &self.model.project,
            &self.model.model,
            version,
        )?;

        let manifest = parse_manifest(version_info.manifest)?;
        let download = self.registry.client.presign_model_download(
            &self.model.namespace,
            &self.model.project,
            &self.model.model,
            version,
        )?;

        let manifest_map = manifest_map(&manifest)?;
        let mut tasks = Vec::new();

        for file in download.files {
            let rel_path = normalize_bundle_path(&file.rel_path);
            let expected = manifest_map.get(&rel_path).ok_or_else(|| {
                RegistryError::InvalidManifest(format!(
                    "download file {rel_path} missing from manifest"
                ))
            })?;

            let dest = safe_join(version_dir, &rel_path)?;
            let needs_download = !file_is_valid(
                &dest,
                expected,
                self.registry.config.verify_checksums,
            )?;

            if needs_download {
                tasks.push(DownloadTask {
                    rel_path,
                    url: file.url,
                    dest,
                    expected: expected.clone(),
                });
            }
        }

        download_tasks(
            &self.registry.http,
            tasks,
            self.registry.config.max_parallel,
        )?;

        write_manifest(version_dir, &manifest)?;
        Ok(CachedModel::new(version_dir.to_path_buf(), manifest))
    }
}

/// Cached model version stored on disk.
pub struct CachedModel {
    root: PathBuf,
    manifest: ModelManifest,
}

impl CachedModel {
    fn new(root: PathBuf, manifest: ModelManifest) -> Self {
        Self { root, manifest }
    }

    /// Root directory of the cached model version.
    pub fn path(&self) -> &Path {
        &self.root
    }

    /// Manifest for the cached model version.
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Build a file-backed bundle reader.
    pub fn reader(&self) -> FsBundleReader {
        FsBundleReader::new(
            self.root.clone(),
            self.manifest
                .files
                .iter()
                .map(|f| f.rel_path.clone())
                .collect(),
        )
    }
}

/// Model version manifest (subset of backend schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Files stored in the bundle.
    pub files: Vec<ManifestFile>,
}

/// Model version file descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestFile {
    /// Path within the bundle.
    pub rel_path: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Checksum (sha256).
    pub checksum: String,
}

/// File-backed bundle reader.
#[derive(Clone)]
pub struct FsBundleReader {
    root: PathBuf,
    files: Vec<String>,
}

impl FsBundleReader {
    /// Create a file-backed bundle reader.
    pub fn new(root: PathBuf, files: Vec<String>) -> Self {
        Self { root, files }
    }
}

impl BundleSource for FsBundleReader {
    fn open(&self, path: &str) -> Result<Box<dyn Read + Send>, String> {
        let rel = sanitize_rel_path(path).map_err(|e| e.to_string())?;
        let full = self.root.join(rel);
        let file = File::open(full).map_err(|e| e.to_string())?;
        Ok(Box::new(file))
    }

    fn list(&self) -> Result<Vec<String>, String> {
        Ok(self.files.clone())
    }
}

const MANIFEST_FILE: &str = "manifest.json";

fn load_manifest(version_dir: &Path) -> Result<ModelManifest, RegistryError> {
    let path = version_dir.join(MANIFEST_FILE);
    let bytes = fs::read(path)?;
    serde_json::from_slice::<ModelManifest>(&bytes)
        .map_err(|e| RegistryError::InvalidManifest(e.to_string()))
}

fn write_manifest(version_dir: &Path, manifest: &ModelManifest) -> Result<(), RegistryError> {
    let path = version_dir.join(MANIFEST_FILE);
    let mut file = File::create(path)?;
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|e| RegistryError::InvalidManifest(e.to_string()))?;
    file.write_all(&bytes)?;
    Ok(())
}

fn parse_manifest(value: serde_json::Value) -> Result<ModelManifest, RegistryError> {
    let mut manifest: ModelManifest = serde_json::from_value(value)
        .map_err(|e| RegistryError::InvalidManifest(e.to_string()))?;

    let mut seen = HashSet::new();
    for file in &mut manifest.files {
        file.rel_path = normalize_bundle_path(&file.rel_path);
        sanitize_rel_path(&file.rel_path)?;
        if file.rel_path.is_empty() {
            return Err(RegistryError::InvalidManifest(
                "manifest file path is empty".to_string(),
            ));
        }
        if !seen.insert(file.rel_path.clone()) {
            return Err(RegistryError::InvalidManifest(format!(
                "duplicate file path in manifest: {}",
                file.rel_path
            )));
        }
    }

    Ok(manifest)
}

fn manifest_map(manifest: &ModelManifest) -> Result<HashMap<String, ManifestFile>, RegistryError> {
    let mut map = HashMap::new();
    for file in &manifest.files {
        if map.insert(file.rel_path.clone(), file.clone()).is_some() {
            return Err(RegistryError::InvalidManifest(format!(
                "duplicate file path in manifest: {}",
                file.rel_path
            )));
        }
    }
    Ok(map)
}

fn cache_is_valid(
    root: &Path,
    manifest: &ModelManifest,
    verify_checksums: bool,
) -> Result<bool, RegistryError> {
    for file in &manifest.files {
        let path = safe_join(root, &file.rel_path)?;
        if !path.exists() {
            return Ok(false);
        }
        let metadata = fs::metadata(&path)?;
        if metadata.len() != file.size_bytes {
            return Ok(false);
        }
        if verify_checksums {
            let (digest, _) = sha256_file(&path)?;
            let expected = normalize_checksum(&file.checksum)?;
            if digest != expected {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

fn file_is_valid(
    path: &Path,
    expected: &ManifestFile,
    verify_checksums: bool,
) -> Result<bool, RegistryError> {
    if !path.exists() {
        return Ok(false);
    }
    let metadata = fs::metadata(path)?;
    if metadata.len() != expected.size_bytes {
        return Ok(false);
    }
    if verify_checksums {
        let (digest, _) = sha256_file(path)?;
        let expected_checksum = normalize_checksum(&expected.checksum)?;
        if digest != expected_checksum {
            return Ok(false);
        }
    }
    Ok(true)
}

fn sha256_file(path: &Path) -> Result<(String, u64), RegistryError> {
    let mut file = File::open(path)?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = [0u8; 1024 * 64];
    let mut total = 0u64;
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
        total += read as u64;
    }
    let digest = format!("{:x}", hasher.finalize());
    Ok((digest, total))
}

fn normalize_checksum(value: &str) -> Result<String, RegistryError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(RegistryError::InvalidManifest(
            "checksum is empty".to_string(),
        ));
    }
    let lower = trimmed.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("sha256:") {
        return Ok(rest.to_string());
    }
    if lower.contains(':') {
        return Err(RegistryError::InvalidManifest(format!(
            "unsupported checksum format: {trimmed}"
        )));
    }
    Ok(lower)
}

fn sanitize_rel_path(path: &str) -> Result<PathBuf, RegistryError> {
    let normalized = normalize_bundle_path(path);
    let rel = Path::new(&normalized);
    for component in rel.components() {
        use std::path::Component;
        match component {
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(RegistryError::InvalidPath(format!(
                    "invalid path component: {path}"
                )))
            }
            Component::CurDir => {
                return Err(RegistryError::InvalidPath(format!(
                    "invalid path component: {path}"
                )))
            }
            Component::Normal(_) => {}
        }
    }
    Ok(PathBuf::from(normalized))
}

fn safe_join(root: &Path, rel: &str) -> Result<PathBuf, RegistryError> {
    let rel = sanitize_rel_path(rel)?;
    Ok(root.join(rel))
}

#[derive(Clone)]
struct DownloadTask {
    rel_path: String,
    url: String,
    dest: PathBuf,
    expected: ManifestFile,
}

fn download_tasks(
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

fn temp_path(dest: &Path) -> Result<PathBuf, RegistryError> {
    let file_name = dest
        .file_name()
        .ok_or_else(|| RegistryError::InvalidPath("missing file name".to_string()))?
        .to_string_lossy();
    Ok(dest.with_file_name(format!(".{file_name}.partial")))
}

/// Helper to check file hashes in the cache (useful for debugging).
#[derive(Debug, Clone)]
pub struct CacheDiagnostics {
    /// Files and their computed checksums.
    pub files: BTreeMap<String, String>,
}

impl CacheDiagnostics {
    /// Compute checksums for a cached model version.
    pub fn from_cached(model: &CachedModel) -> Result<Self, RegistryError> {
        let mut files = BTreeMap::new();
        for file in &model.manifest.files {
            let path = safe_join(model.path(), &file.rel_path)?;
            let (digest, _) = sha256_file(&path)?;
            files.insert(file.rel_path.clone(), digest);
        }
        Ok(Self { files })
    }
}
