//! This crate centralizes traits, structures and utilities for handling artifacts in Burn Central.

mod tools;
mod transfer;

pub mod bundle;
pub mod download;
pub mod upload;

pub use tools::validation::normalize_checksum;
pub use transfer::{FileTransferClient, ReqwestTransferClient};
