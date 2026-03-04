//! This module defines the core traits and structures for working with bundles of artifact files in Burn Central.
//!
//! Bundles are a way to group multiple related files together as a single artifact, with support for encoding/decoding complex data structures into bundles of files, and abstracting over different storage backends for reading/writing those bundles.
//!
//! As a user of Burn Central, you will typically interact with bundles indirectly through higher-level APIs for logging experiment artifacts, registering models, etc.
//! However, if you need to implement custom artifact handling logic (e.g. for a new model format), you may need to implement the BundleEncode/BundleDecode traits for your data structures, and use the BundleSink/BundleSource traits to read/write files from/to bundles in a storage-agnostic way.
//!
//! # Examples
//!
//! ```
//! use burn_central_artifact::bundle::{BundleEncode, BundleDecode, BundleSink, BundleSource};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct MyModel {
//!     name: String,
//!     parameters: Vec<f32>,
//! }!
//! impl BundleEncode for MyModel {
//!     type Settings = ();
//!     type Error = String;
//!     fn encode<O: BundleSink>(self, sink: &mut O, _settings: &Self::Settings) -> Result<(), Self::Error> {
//!         let json = serde_json::to_string(&self).map_err(|e| e.to_string())?;
//!         sink.put_bytes("model.json", json.as_bytes()).map_err(|e| e.to_string())?;
//!         Ok(())
//!     }
//! }
//! impl BundleDecode for MyModel {
//!     type Settings = ();
//!     type Error = String;
//!     fn decode<I: BundleSource>(source: &I, _settings: &Self::Settings) -> Result<Self, Self::Error> {
//!         let mut reader = source.open("model.json").map_err(|e| e.to_string())?;
//!         let mut json = String::new();
//!         reader.read_to_string(&mut json).map_err(|e| e.to_string())?;
//!         serde_json::from_str(&json).map_err(|e| e.to_string())
//!     }
//! }
//! ```

mod fs;
mod memory;

pub use fs::*;
pub use memory::*;

use serde::{Serialize, de::DeserializeOwned};
use std::io::Read;

/// Trait for encoding data into a bundle of files
///
/// Implementors should write their data to the provided BundleSink, which abstracts over the underlying storage mechanism. The Settings associated type can be used to pass any necessary configuration for encoding (e.g. compression level, file naming conventions, etc).
pub trait BundleEncode {
    /// Settings type for encoding, which can include any necessary configuration (e.g. compression level, file naming conventions, etc).
    type Settings: Default + Serialize + DeserializeOwned;
    /// Error type for encoding failures. Should be convertible to a generic error type for ease of use in higher-level APIs.
    type Error: Into<Box<dyn std::error::Error + Send + Sync + 'static>>;

    /// Encode the data into the provided BundleSink. The sink should be used to write all files that are part of the bundle, and the implementation should return an error if encoding fails for any reason (e.g. serialization errors, I/O errors, etc).
    fn encode<O: BundleSink>(
        self,
        sink: &mut O,
        settings: &Self::Settings,
    ) -> Result<(), Self::Error>;
}

/// Trait for decoding data from a bundle of files
///
/// Implementors should read their data from the provided BundleSource, which abstracts over the underlying storage mechanism. The Settings associated type can be used to pass any necessary configuration for decoding (e.g. expected file names, compression settings, etc).
pub trait BundleDecode: Sized {
    /// Settings type for decoding, which can include any necessary configuration (e.g. expected file names, compression settings, etc).
    type Settings: Default + Serialize + DeserializeOwned;
    /// Error type for decoding failures. Should be convertible to a generic error type for ease of use in higher-level APIs.
    type Error: Into<Box<dyn std::error::Error + Send + Sync + 'static>>;

    /// Decode the data from the provided BundleSource. The implementation should return an error if decoding fails for any reason (e.g. deserialization errors, I/O errors, etc).
    fn decode<I: BundleSource>(source: &I, settings: &Self::Settings) -> Result<Self, Self::Error>;
}

/// Trait for writing files to a bundle
pub trait BundleSink {
    /// Add a file by streaming its bytes. Returns computed checksum + size.
    fn put_file<R: Read>(&mut self, path: &str, reader: &mut R) -> Result<(), String>;

    /// Convenience: write all bytes.
    fn put_bytes(&mut self, path: &str, bytes: &[u8]) -> Result<(), String> {
        let mut r = std::io::Cursor::new(bytes);
        self.put_file(path, &mut r)
    }
}

/// Trait for reading files from a bundle
pub trait BundleSource {
    /// Open the given path for streaming read. Must validate existence.
    fn open(&self, path: &str) -> Result<Box<dyn Read + Send>, String>;

    /// Optionally list available files (used by generic decoders; can be best-effort).
    fn list(&self) -> Result<Vec<String>, String>;
}
