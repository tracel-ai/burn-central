use std::collections::BTreeMap;

use crate::cache::{safe_join, sha256_file};
use crate::error::RegistryError;
use crate::model::CachedModel;

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
        for file in &model.manifest().files {
            let path = safe_join(model.path(), &file.rel_path)?;
            let (digest, _) = sha256_file(&path)?;
            files.insert(file.rel_path.clone(), digest);
        }
        Ok(Self { files })
    }
}
