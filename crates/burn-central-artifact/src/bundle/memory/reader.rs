use std::collections::BTreeMap;
use std::io::Read;

use crate::bundle::BundleSource;
use crate::tools::path::normalize_bundle_path;

/// In-memory reader for synthetic or cached bundles.
pub struct InMemoryBundleReader {
    files: BTreeMap<String, Vec<u8>>, // rel_path -> bytes
}

impl InMemoryBundleReader {
    pub fn new(files: BTreeMap<String, Vec<u8>>) -> Self {
        Self { files }
    }

    /// Get the files in this bundle
    pub fn files(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.files
    }

    /// Check if a file exists in the bundle
    pub fn contains_file(&self, path: &str) -> bool {
        let normalized = normalize_bundle_path(path);
        self.files.contains_key(&normalized)
    }

    /// Get the size of a file in the bundle
    pub fn file_size(&self, path: &str) -> Option<usize> {
        let normalized = normalize_bundle_path(path);
        self.files.get(&normalized).map(|bytes| bytes.len())
    }

    /// Get all file paths in the bundle
    pub fn file_paths(&self) -> Vec<String> {
        self.files.keys().cloned().collect()
    }
}

impl BundleSource for InMemoryBundleReader {
    fn open(&self, path: &str) -> Result<Box<dyn Read + Send>, String> {
        let rel = normalize_bundle_path(path);
        let bytes = self
            .files
            .get(&rel)
            .ok_or_else(|| format!("File not found in bundle: {}", rel))?;
        Ok(Box::new(std::io::Cursor::new(bytes.clone())))
    }

    fn list(&self) -> Result<Vec<String>, String> {
        Ok(self.files.keys().cloned().collect())
    }
}
