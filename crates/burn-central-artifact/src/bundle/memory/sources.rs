use std::io::Read;

use crate::{bundle::BundleSink, tools::path::normalize_bundle_path};

/// A builder for creating bundles with multiple files
#[derive(Default, Clone)]
pub struct InMemoryBundleSources {
    files: Vec<PendingFile>,
}

impl InMemoryBundleSources {
    /// Create a new empty bundle sources
    pub fn new() -> Self {
        Self::default()
    }

    /// Add raw bytes as a file within the bundle at `dest_path`.
    pub fn add_bytes(mut self, bytes: Vec<u8>, dest_path: impl AsRef<str>) -> Self {
        self.files.push(PendingFile {
            dest_path: normalize_bundle_path(dest_path.as_ref()),
            source: bytes,
        });
        self
    }

    /// Add a file from a reader
    pub fn add_file<R: Read>(
        self,
        mut reader: R,
        dest_path: impl AsRef<str>,
    ) -> Result<Self, std::io::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Ok(self.add_bytes(bytes, dest_path))
    }

    /// Get the files in this bundle sources
    pub fn files(&self) -> &Vec<PendingFile> {
        &self.files
    }

    /// Convert into the files vector
    pub fn into_files(self) -> Vec<PendingFile> {
        self.files
    }

    /// Check if the bundle is empty
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Get the number of files
    pub fn len(&self) -> usize {
        self.files.len()
    }
}

impl BundleSink for InMemoryBundleSources {
    fn put_file<R: Read>(&mut self, path: &str, reader: &mut R) -> Result<(), String> {
        let mut buf = Vec::new();
        reader
            .read_to_end(&mut buf)
            .map_err(|e| format!("Failed to read from source: {}", e))?;
        self.files.push(PendingFile {
            dest_path: normalize_bundle_path(path),
            source: buf,
        });
        Ok(())
    }
}

/// A file that is pending to be added to a bundle
#[derive(Clone)]
pub struct PendingFile {
    pub dest_path: String, // path within the bundle (use forward slashes)
    pub source: Vec<u8>,
}

impl PendingFile {
    /// Get the destination path of this file
    pub fn dest_path(&self) -> &str {
        &self.dest_path
    }

    /// Get the source bytes of this file
    pub fn source(&self) -> &[u8] {
        &self.source
    }

    /// Get the size of this file
    pub fn size(&self) -> usize {
        self.source.len()
    }
}
