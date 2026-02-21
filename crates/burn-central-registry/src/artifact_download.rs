use std::path::Path;

use burn_central_core::bundle::normalize_bundle_path;

use crate::RegistryError;
use crate::cache::safe_join;
use crate::download::{DownloadTask, download_tasks};
use crate::manifest::ManifestFile;

/// Generic download descriptor for any model artifact file.
#[derive(Debug, Clone)]
pub struct ArtifactDownloadFile {
    pub rel_path: String,
    pub url: String,
    pub size_bytes: u64,
    pub checksum: String,
}

/// Download artifact files into a destination directory, validating size and checksum.
pub fn download_artifacts_to_dir(
    dest_root: &Path,
    files: &[ArtifactDownloadFile],
) -> Result<(), RegistryError> {
    std::fs::create_dir_all(dest_root)?;

    if files.is_empty() {
        return Ok(());
    }

    let mut tasks = Vec::with_capacity(files.len());
    for file in files {
        let rel_path = normalize_bundle_path(&file.rel_path);
        let dest = safe_join(dest_root, &rel_path)?;

        tasks.push(DownloadTask {
            rel_path: rel_path.clone(),
            url: file.url.clone(),
            dest,
            expected: ManifestFile {
                rel_path,
                size_bytes: file.size_bytes,
                checksum: file.checksum.clone(),
            },
        });
    }

    let parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let http = reqwest::blocking::Client::new();
    download_tasks(&http, tasks, parallelism)
}
