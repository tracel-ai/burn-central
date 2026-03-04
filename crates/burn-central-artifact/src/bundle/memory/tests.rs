//! Example of how to use the bundle abstractions
//!
//! This module shows how to implement BundleEncode and BundleDecode
//! for custom data types and use them with artifacts and models.

use serde::{Deserialize, Serialize};
use std::io::Read;

use crate::bundle::{BundleDecode, BundleEncode, BundleSink, BundleSource};

/// Example configuration that can be encoded as a bundle
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExampleConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub model_name: String,
    pub features: Vec<String>,
}

/// Settings for encoding/decoding ExampleConfig
#[derive(Debug, Serialize, Deserialize)]
pub struct ExampleConfigSettings {
    pub config_filename: String,
    pub metadata_filename: String,
}

impl Default for ExampleConfigSettings {
    fn default() -> Self {
        Self {
            config_filename: "config.json".to_string(),
            metadata_filename: "metadata.json".to_string(),
        }
    }
}

/// Example metadata to accompany the config
#[derive(Debug, Serialize, Deserialize)]
struct ConfigMetadata {
    created_at: String,
    version: String,
    description: String,
}

impl BundleEncode for ExampleConfig {
    type Settings = ExampleConfigSettings;
    type Error = String;

    fn encode<O: BundleSink>(
        self,
        sink: &mut O,
        settings: &Self::Settings,
    ) -> Result<(), Self::Error> {
        // Encode the main config as JSON
        let config_json = serde_json::to_string_pretty(&self)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        sink.put_bytes(&settings.config_filename, config_json.as_bytes())
            .map_err(|e| format!("Failed to write config file: {}", e))?;

        // Create and encode metadata
        let metadata = ConfigMetadata {
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
            version: "1.0.0".to_string(),
            description: format!("Configuration for model: {}", self.model_name),
        };

        let metadata_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| format!("Failed to serialize metadata: {}", e))?;

        sink.put_bytes(&settings.metadata_filename, metadata_json.as_bytes())
            .map_err(|e| format!("Failed to write metadata file: {}", e))?;

        Ok(())
    }
}

impl BundleDecode for ExampleConfig {
    type Settings = ExampleConfigSettings;
    type Error = String;

    fn decode<I: BundleSource>(source: &I, settings: &Self::Settings) -> Result<Self, Self::Error> {
        // Read the config file
        let mut config_reader = source
            .open(&settings.config_filename)
            .map_err(|e| format!("Failed to open config file: {}", e))?;

        let mut config_content = String::new();
        config_reader
            .read_to_string(&mut config_content)
            .map_err(|e| format!("Failed to read config file: {}", e))?;

        let config: ExampleConfig = serde_json::from_str(&config_content)
            .map_err(|e| format!("Failed to parse config JSON: {}", e))?;

        // Optionally validate metadata exists (but don't require it for decoding)
        if let Ok(mut metadata_reader) = source.open(&settings.metadata_filename) {
            let mut metadata_content = String::new();
            if metadata_reader
                .read_to_string(&mut metadata_content)
                .is_ok()
            {
                if let Ok(_metadata) = serde_json::from_str::<ConfigMetadata>(&metadata_content) {
                    // Could validate version compatibility here
                }
            }
        }

        Ok(config)
    }
}

use crate::bundle::{InMemoryBundleReader, InMemoryBundleSources};
use std::collections::BTreeMap;

#[test]
fn test_config_encode_decode() {
    let original_config = ExampleConfig {
        learning_rate: 0.001,
        batch_size: 32,
        model_name: "transformer".to_string(),
        features: vec!["attention".to_string(), "dropout".to_string()],
    };

    let settings = ExampleConfigSettings::default();

    // Encode to bundle
    let mut sources = InMemoryBundleSources::new();
    original_config
        .clone()
        .encode(&mut sources, &settings)
        .expect("Encoding should succeed");

    // Convert to memory reader for decoding
    let mut file_map = BTreeMap::new();
    for file in sources.into_files() {
        file_map.insert(file.dest_path().to_string(), file.source().to_vec());
    }
    let reader = InMemoryBundleReader::new(file_map);

    // Decode from bundle
    let decoded_config =
        ExampleConfig::decode(&reader, &settings).expect("Decoding should succeed");

    assert_eq!(original_config, decoded_config);
}

#[test]
fn test_bundle_contains_expected_files() {
    let config = ExampleConfig {
        learning_rate: 0.001,
        batch_size: 32,
        model_name: "test_model".to_string(),
        features: vec!["feature1".to_string()],
    };

    let settings = ExampleConfigSettings::default();
    let mut sources = InMemoryBundleSources::new();
    config.encode(&mut sources, &settings).unwrap();

    let files: Vec<String> = sources
        .files()
        .iter()
        .map(|f| f.dest_path().to_string())
        .collect();

    assert!(files.contains(&settings.config_filename));
    assert!(files.contains(&settings.metadata_filename));
    assert_eq!(files.len(), 2);
}
