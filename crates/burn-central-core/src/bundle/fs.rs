use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::Digest;
use tempfile::TempDir;

use crate::bundle::{BundleSink, BundleSource, normalize_bundle_path};

/// File-backed bundle sink that streams files to disk and computes checksums.
#[derive(Debug)]
pub struct FsBundleSink {
    root: PathBuf,
    files: Vec<FsBundleFile>,
    seen: HashSet<String>,
    #[allow(unused)]
    _temp: Option<TempDir>,
}

impl FsBundleSink {
    /// Create a bundle sink rooted at the provided directory.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, std::io::Error> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            files: Vec::new(),
            seen: HashSet::new(),
            _temp: None,
        })
    }

    /// Create a temporary bundle sink that cleans up on drop.
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

    /// Root directory for the bundle files.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Files written into the bundle.
    pub fn files(&self) -> &[FsBundleFile] {
        &self.files
    }

    /// Consume the sink and return the file list.
    pub fn into_files(self) -> Vec<FsBundleFile> {
        self.files
    }
}

impl BundleSink for FsBundleSink {
    fn put_file<R: Read>(&mut self, path: &str, reader: &mut R) -> Result<(), String> {
        let rel = sanitize_rel_path(path)?;

        if !self.seen.insert(rel.to_string()) {
            return Err(format!("Duplicate bundle path: {rel}"));
        }

        let dest = self.root.join(&rel);
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
            rel_path: rel.to_string(),
            abs_path: dest,
            size_bytes: total,
            checksum,
        });

        Ok(())
    }
}

/// File descriptor emitted by a file-backed bundle sink.
#[derive(Debug, Clone)]
pub struct FsBundleFile {
    /// Relative path within the bundle.
    pub rel_path: String,
    /// Absolute file system path for the cached file.
    pub abs_path: PathBuf,
    /// Size in bytes.
    pub size_bytes: u64,
    /// SHA-256 checksum (hex).
    pub checksum: String,
}

/// File-backed bundle reader for streaming decode.
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
        let rel = sanitize_rel_path(path)?;
        let full = self.root.join(rel);
        let file = File::open(full).map_err(|e| e.to_string())?;
        Ok(Box::new(file))
    }

    fn list(&self) -> Result<Vec<String>, String> {
        Ok(self.files.clone())
    }
}

fn sanitize_rel_path(path: &str) -> Result<String, String> {
    let normalized = normalize_bundle_path(path);
    if normalized.is_empty() {
        return Err("Empty bundle path".to_string());
    }

    let rel = Path::new(&normalized);
    for component in rel.components() {
        use std::path::Component;
        match component {
            Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_)
            | Component::CurDir => {
                return Err(format!("Invalid bundle path: {path}"));
            }
            Component::Normal(_) => {}
        }
    }

    Ok(normalized)
}

fn temp_path(dest: &Path) -> Result<PathBuf, std::io::Error> {
    let file_name = dest
        .file_name()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Missing file name"))?
        .to_string_lossy();
    Ok(dest.with_file_name(format!(".{file_name}.partial")))
}
