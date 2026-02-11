use std::fs;
use std::path::{Path, PathBuf};

use burn_central_core::bundle::{BundleDecode, FsBundleReader, normalize_bundle_path};

use crate::cache::{cache_is_valid, file_is_valid, safe_join};
use crate::download::{DownloadTask, download_tasks};
use crate::error::RegistryError;
use crate::manifest::{ModelManifest, load_manifest, manifest_map, parse_manifest, write_manifest};
use crate::registry::Registry;

/// Selector for which model version to load.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelVersionSelector {
    Latest,
    Version(u64),
}

/// Type alias for model version numbers.
pub type ModelVersion = u64;

impl From<u32> for ModelVersionSelector {
    fn from(value: u32) -> Self {
        ModelVersionSelector::Version(value as u64)
    }
}

impl From<u64> for ModelVersionSelector {
    fn from(value: u64) -> Self {
        ModelVersionSelector::Version(value)
    }
}

impl Default for ModelVersionSelector {
    fn default() -> Self {
        ModelVersionSelector::Latest
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
    pub fn new(
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

    fn version_dir(&self, root: &Path, version: ModelVersion) -> PathBuf {
        root.join(&self.namespace)
            .join(&self.project)
            .join("models")
            .join(&self.model)
            .join("versions")
            .join(version.to_string())
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn project(&self) -> &str {
        &self.project
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Validate that a path component doesn't contain directory traversal characters.
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
                )));
            }
            Component::Normal(_) => {}
            Component::CurDir => {
                return Err(RegistryError::InvalidPath(format!(
                    "{label} contains invalid path segments"
                )));
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
    pub fn new(registry: Registry, model: ModelRef) -> Self {
        Self { registry, model }
    }

    /// Ensure a model version is cached locally.
    pub fn ensure(
        &self,
        version: impl Into<ModelVersionSelector>,
    ) -> Result<CachedModel, RegistryError> {
        let version = match version.into() {
            ModelVersionSelector::Latest => {
                let info = self.registry.client().get_model(
                    self.model.namespace(),
                    self.model.project(),
                    self.model.model(),
                )?;
                info.version_count - 1
            }
            ModelVersionSelector::Version(v) => v,
        };
        let version_dir = self.model.version_dir(&self.registry.cache_dir(), version);

        if let Ok(manifest) = load_manifest(&version_dir) {
            if cache_is_valid(&version_dir, &manifest, true)? {
                return Ok(CachedModel::new(version_dir, manifest));
            }
        }

        self.download_version(version, &version_dir)
    }

    /// Download and decode a model version using the bundle decoder.
    pub fn load<T: BundleDecode>(
        &self,
        version: impl Into<ModelVersionSelector>,
        settings: &T::Settings,
    ) -> Result<T, RegistryError> {
        let cached = self.ensure(version)?;
        T::decode(&cached.reader(), settings)
            .map_err(|e| RegistryError::Decode(e.into().to_string()))
    }

    fn download_version(
        &self,
        version: ModelVersion,
        version_dir: &Path,
    ) -> Result<CachedModel, RegistryError> {
        fs::create_dir_all(version_dir)?;

        let version_info = self.registry.client().get_model_version(
            self.model.namespace(),
            self.model.project(),
            self.model.model(),
            version as _,
        )?;

        let manifest = parse_manifest(version_info.manifest)?;
        let download = self.registry.client().presign_model_download(
            self.model.namespace(),
            self.model.project(),
            self.model.model(),
            version as _,
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
            let needs_download = !file_is_valid(&dest, expected, true)?;

            if needs_download {
                tasks.push(DownloadTask {
                    rel_path,
                    url: file.url,
                    dest,
                    expected: expected.clone(),
                });
            }
        }

        let parallelism = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        download_tasks(self.registry.http(), tasks, parallelism)?;

        write_manifest(version_dir, &manifest)?;
        Ok(CachedModel::new(version_dir.to_path_buf(), manifest))
    }
}

/// Cached model version stored on disk.
#[derive(Debug, Clone)]
pub struct CachedModel {
    root: PathBuf,
    manifest: ModelManifest,
}

impl CachedModel {
    pub fn new(root: PathBuf, manifest: ModelManifest) -> Self {
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
