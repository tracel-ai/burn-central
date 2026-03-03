//! This crate centralizes traits, structures and utilities for handling artifacts and models in Burn Central.

mod artifact_download;
mod download;
mod tools;

pub use artifact_download::{ArtifactDownloadFile, download_artifacts_to_dir};
pub use download::DownloadError;
