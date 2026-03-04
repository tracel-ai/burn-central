//! File-backed bundle implementation for Burn Central artifacts.
//!
//! This module provides an implementation of the `BundleSink` and `BundleSource` traits that uses the local filesystem to store artifact files.
//! It supports both temporary bundles (which clean up after themselves) and persistent bundles rooted at a specified directory.
//! The implementation ensures that file paths are sanitized to prevent directory traversal, and that concurrent writes to the same path are handled safely using temporary files and atomic renames.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use sha2::Digest;
use tempfile::TempDir;

use crate::bundle::{BundleSink, BundleSource};
use crate::tools::path::{safe_join, sanitize_rel_path};
use crate::upload::{MultipartUploadSource, UploadError};

/// File-backed bundle that can both read and write artifact files.
#[derive(Debug)]
pub struct FsBundle {
    root: PathBuf,
    files: Vec<FsBundleFile>,
    seen: HashSet<String>,
    _temp: Option<TempDir>,
}

impl FsBundle {
    /// Create a writable bundle rooted at the provided directory.
    pub fn create(root: impl Into<PathBuf>) -> Result<Self, std::io::Error> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            files: Vec::new(),
            seen: HashSet::new(),
            _temp: None,
        })
    }

    /// Create a temporary writable bundle that cleans up on drop.
    pub fn temp() -> Result<Self, std::io::Error> {
        let temp = TempDir::new()?;
        let root = temp.path().to_path_buf();
        Ok(Self {
            root,
            files: Vec::new(),
            seen: HashSet::new(),
            _temp: Some(temp),
        })
    }

    /// Create a read-oriented bundle backed by an existing root + file list.
    pub fn with_files(root: PathBuf, files: Vec<String>) -> Result<Self, String> {
        let mut bundle = Self {
            root,
            files: Vec::new(),
            seen: HashSet::new(),
            _temp: None,
        };
        for path in files {
            bundle.register_file(&path)?;
        }
        Ok(bundle)
    }

    /// Root directory for the bundle files.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Files written into the bundle.
    pub fn files(&self) -> &[FsBundleFile] {
        &self.files
    }

    /// Relative file paths currently indexed by this bundle.
    pub fn file_paths(&self) -> Vec<String> {
        self.files.iter().map(|f| f.rel_path.clone()).collect()
    }

    /// Register an existing file path in the bundle, ensuring it is valid and not duplicated.
    fn register_file(&mut self, path: &str) -> Result<(), String> {
        let rel = sanitize_rel_path(path)?.to_string_lossy().to_string();
        if rel.is_empty() || !self.seen.insert(rel.clone()) {
            return Err(format!("Duplicate bundle path: {rel}"));
        }

        self.files.push(FsBundleFile {
            rel_path: rel.clone(),
            abs_path: self.root.join(&rel),
            size_bytes: None,
            checksum: None,
        });

        Ok(())
    }

    /// Delete all files in this bundle from the filesystem. This is idempotent and can be used for cleanup.
    pub fn delete(self) -> Result<(), std::io::Error> {
        for file in &self.files {
            let path = safe_join(&self.root, &file.rel_path);
            if let Ok(path) = path {
                match fs::remove_file(path) {
                    Ok(()) => {}
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                    Err(err) => return Err(err),
                }
            }
        }
        Ok(())
    }

    fn clear_temp_files(&self) {
        for file in &self.files {
            let tmp = temp_path(&file.abs_path);
            if let Ok(tmp) = tmp {
                if tmp.exists() {
                    let _ = fs::remove_file(tmp);
                }
            }
        }
    }
}

impl Drop for FsBundle {
    fn drop(&mut self) {
        self.clear_temp_files();
    }
}

impl BundleSink for FsBundle {
    fn put_file<R: Read>(&mut self, path: &str, reader: &mut R) -> Result<(), String> {
        let rel = sanitize_rel_path(path).map_err(|e| e.to_string())?;
        let rel = rel.to_string_lossy().to_string();

        if !self.seen.insert(rel.clone()) {
            return Err(format!("Duplicate bundle path: {rel}"));
        }

        let dest = safe_join(&self.root, &rel).map_err(|e| e.to_string())?;
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }

        let tmp = temp_path(&dest).map_err(|e| e.to_string())?;
        let mut file = File::create(&tmp).map_err(|e| e.to_string())?;

        let mut hasher = sha2::Sha256::new();
        let mut buf = [0u8; 1024 * 64];
        let mut total = 0u64;

        loop {
            let read = reader.read(&mut buf).map_err(|e| e.to_string())?;
            if read == 0 {
                break;
            }
            file.write_all(&buf[..read]).map_err(|e| e.to_string())?;
            hasher.update(&buf[..read]);
            total += read as u64;
        }

        let checksum = format!("{:x}", hasher.finalize());

        if dest.exists() {
            fs::remove_file(&dest).map_err(|e| e.to_string())?;
        }

        fs::rename(&tmp, &dest).map_err(|e| e.to_string())?;

        self.files.push(FsBundleFile {
            rel_path: rel,
            abs_path: dest,
            size_bytes: Some(total),
            checksum: Some(checksum),
        });

        Ok(())
    }
}

/// File descriptor emitted by a file-backed bundle.
#[derive(Debug, Clone)]
pub struct FsBundleFile {
    /// Relative path within the bundle.
    pub rel_path: String,
    /// Absolute file system path for the cached file.
    pub abs_path: PathBuf,
    /// Size in bytes, when known.
    pub size_bytes: Option<u64>,
    /// SHA-256 checksum (hex), when known.
    pub checksum: Option<String>,
}

impl BundleSource for FsBundle {
    fn open(&self, path: &str) -> Result<Box<dyn Read + Send>, String> {
        let rel = sanitize_rel_path(path).map_err(|e| e.to_string())?;
        let rel = rel.to_string_lossy().to_string();

        if !self.seen.contains(&rel) {
            return Err(format!("Bundle path not found: {rel}"));
        }

        let file_path = safe_join(&self.root, &rel).map_err(|e| e.to_string())?;
        let file = File::open(&file_path).map_err(|e| e.to_string())?;
        Ok(Box::new(file))
    }

    fn list(&self) -> Result<Vec<String>, String> {
        Ok(self.file_paths())
    }
}

fn temp_path(dest: &Path) -> Result<PathBuf, std::io::Error> {
    let file_name = dest
        .file_name()
        .ok_or_else(|| std::io::Error::other("Missing file name"))?
        .to_string_lossy();
    Ok(dest.with_file_name(format!(".{file_name}.partial")))
}

impl MultipartUploadSource for FsBundle {
    fn file_len(&self, rel_path: &str) -> Result<u64, UploadError> {
        let source = safe_join(self.root(), rel_path).map_err(UploadError::InvalidMultipart)?;
        let metadata = std::fs::metadata(&source).map_err(|e| {
            UploadError::InvalidMultipart(format!(
                "Missing file for multipart upload {}: {}",
                rel_path, e
            ))
        })?;
        if !metadata.is_file() {
            return Err(UploadError::InvalidMultipart(format!(
                "Multipart upload source is not a file: {}",
                rel_path
            )));
        }

        Ok(metadata.len())
    }

    fn open_part(
        &self,
        rel_path: &str,
        offset: u64,
        size: u64,
    ) -> Result<Box<dyn Read + Send>, UploadError> {
        let source = safe_join(self.root(), rel_path).map_err(UploadError::InvalidMultipart)?;
        let mut file = File::open(&source).map_err(|e| UploadError::MultipartReader {
            rel_path: rel_path.to_string(),
            source: Box::new(e),
        })?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| UploadError::MultipartReader {
                rel_path: rel_path.to_string(),
                source: Box::new(e),
            })?;
        Ok(Box::new(file.take(size)))
    }
}
