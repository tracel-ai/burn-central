use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use burn_central_core::bundle::normalize_bundle_path;
use sha2::Digest;

use crate::error::RegistryError;
use crate::manifest::{ManifestFile, ModelManifest};

/// Check if a cached model version is valid.
pub fn cache_is_valid(
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

/// Check if a single file is valid according to its manifest entry.
pub fn file_is_valid(
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

/// Compute SHA256 checksum of a file.
pub fn sha256_file(path: &Path) -> Result<(String, u64), RegistryError> {
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

/// Normalize a checksum string (strip prefixes, lowercase).
pub fn normalize_checksum(value: &str) -> Result<String, RegistryError> {
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

/// Sanitize a relative path to prevent directory traversal attacks.
pub fn sanitize_rel_path(path: &str) -> Result<PathBuf, RegistryError> {
    let normalized = normalize_bundle_path(path);
    let rel = Path::new(&normalized);
    for component in rel.components() {
        use std::path::Component;
        match component {
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(RegistryError::InvalidPath(format!(
                    "invalid path component: {path}"
                )));
            }
            Component::CurDir => {
                return Err(RegistryError::InvalidPath(format!(
                    "invalid path component: {path}"
                )));
            }
            Component::Normal(_) => {}
        }
    }
    Ok(PathBuf::from(normalized))
}

/// Safely join a root path with a relative path.
pub fn safe_join(root: &Path, rel: &str) -> Result<PathBuf, RegistryError> {
    let rel = sanitize_rel_path(rel)?;
    Ok(root.join(rel))
}
