use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use burn_central_core::bundle::normalize_bundle_path;
use serde::{Deserialize, Serialize};

use crate::cache::sanitize_rel_path;
use crate::error::RegistryError;

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

const MANIFEST_FILE: &str = "manifest.json";

/// Load manifest from a cached version directory.
pub fn load_manifest(version_dir: &Path) -> Result<ModelManifest, RegistryError> {
    let path = version_dir.join(MANIFEST_FILE);
    let bytes = fs::read(path)?;
    serde_json::from_slice::<ModelManifest>(&bytes)
        .map_err(|e| RegistryError::InvalidManifest(e.to_string()))
}

/// Write manifest to a cached version directory.
pub fn write_manifest(version_dir: &Path, manifest: &ModelManifest) -> Result<(), RegistryError> {
    let path = version_dir.join(MANIFEST_FILE);
    let mut file = File::create(path)?;
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|e| RegistryError::InvalidManifest(e.to_string()))?;
    file.write_all(&bytes)?;
    Ok(())
}

/// Parse and validate a manifest from JSON.
pub fn parse_manifest(value: serde_json::Value) -> Result<ModelManifest, RegistryError> {
    let mut manifest: ModelManifest =
        serde_json::from_value(value).map_err(|e| RegistryError::InvalidManifest(e.to_string()))?;

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

/// Create a hashmap of manifest files by relative path.
pub fn manifest_map(
    manifest: &ModelManifest,
) -> Result<HashMap<String, ManifestFile>, RegistryError> {
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
