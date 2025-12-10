use std::path::PathBuf;

use crate::artifacts::ArtifactKind;
use crate::bundle::{BundleDecode, BundleEncode, BundleSink};
use crate::experiment::{ExperimentRun, ExperimentRunHandle};
use crate::burn::record::{
    FileRecorder, FullPrecisionSettings, NamedMpkBytesRecorder, Record, Recorder, RecorderError,
};
use crate::burn::tensor::backend::Backend;
use serde::Deserialize;
use serde::{Serialize, de::DeserializeOwned};

struct CheckpointRecordSources<B, R> {
    pub backend: std::marker::PhantomData<B>,
    pub record: R,
}

impl<B, R> CheckpointRecordSources<B, R> {
    pub fn new(record: R) -> Self {
        Self {
            backend: std::marker::PhantomData,
            record,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct CheckpointRecordArtifactSettings {
    pub name: String,
}

impl Default for CheckpointRecordArtifactSettings {
    fn default() -> Self {
        Self {
            name: "checkpoint.mpk".to_string(),
        }
    }
}

impl<B, R> BundleEncode for CheckpointRecordSources<B, R>
where
    R: Record<B>,
    B: Backend,
{
    type Settings = CheckpointRecordArtifactSettings;
    type Error = String;
    fn encode<E: BundleSink>(
        self,
        sink: &mut E,
        settings: &Self::Settings,
    ) -> Result<(), Self::Error> {
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = recorder
            .record(self.record, ())
            .map_err(|e| format!("Failed to record to bytes: {}", e))?;
        sink.put_bytes(&settings.name, &bytes)
            .map_err(|e| format!("Failed to write bytes to sink: {}", e))?;
        Ok(())
    }
}

impl<B, R> BundleDecode for CheckpointRecordSources<B, R>
where
    R: Record<B>,
    B: Backend,
{
    type Settings = CheckpointRecordArtifactSettings;
    type Error = String;

    fn decode<I: crate::bundle::BundleSource>(
        source: &I,
        settings: &Self::Settings,
    ) -> Result<Self, Self::Error> {
        let mut reader = source.open(&settings.name).map_err(|e| {
            format!(
                "Failed to get reader for checkpoint artifact {}: {}",
                settings.name, e
            )
        })?;
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).map_err(|e| {
            format!(
                "Failed to read bytes for checkpoint artifact {}: {}",
                settings.name, e
            )
        })?;
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let record = recorder
            .load::<R>(bytes, &B::Device::default())
            .map_err(|e| format!("Failed to load record from bytes: {}", e))?;
        Ok(Self::new(record))
    }
}

/// A recorder that saves and loads data from a remote server using the [BurnCentralClientState](BurnCentralClientState).
#[derive(Debug, Clone)]
pub struct RemoteCheckpointRecorder {
    experiment_handle: ExperimentRunHandle,
}

impl RemoteCheckpointRecorder {
    pub fn new(experiment: &ExperimentRun) -> Self {
        Self {
            experiment_handle: experiment.handle(),
        }
    }
}

impl<B: Backend> FileRecorder<B> for RemoteCheckpointRecorder {
    fn file_extension() -> &'static str {
        "mpk"
    }
}

impl Default for RemoteCheckpointRecorder {
    fn default() -> Self {
        unimplemented!("Default is not implemented for RemoteRecorder, as it requires a client.")
    }
}

impl<B: Backend> Recorder<B> for RemoteCheckpointRecorder {
    type Settings = FullPrecisionSettings;
    type RecordArgs = PathBuf;
    type RecordOutput = ();
    type LoadArgs = PathBuf;

    fn record<R>(
        &self,
        record: R,
        args: Self::RecordArgs,
    ) -> Result<Self::RecordOutput, RecorderError>
    where
        R: Record<B>,
    {
        let file_name = args
            .file_name()
            .ok_or(RecorderError::Unknown(
                "File name should be present".to_string(),
            ))?
            .to_str()
            .ok_or(RecorderError::Unknown(
                "File name should be a valid string".to_string(),
            ))?;
        let settings = CheckpointRecordArtifactSettings {
            name: file_name.to_string(),
        };
        self.experiment_handle
            .log_artifact(
                file_name,
                ArtifactKind::Other,
                CheckpointRecordSources::new(record),
                &settings,
            )
            .map_err(|e| RecorderError::Unknown(format!("Failed to record artifact: {e}")))
    }

    fn load<R>(&self, args: Self::LoadArgs, _device: &B::Device) -> Result<R, RecorderError>
    where
        R: Record<B>,
    {
        let name = args
            .file_name()
            .ok_or(RecorderError::Unknown(
                "File name should be present".to_string(),
            ))?
            .to_str()
            .ok_or(RecorderError::Unknown(
                "File name should be a valid string".to_string(),
            ))?;

        let settings = CheckpointRecordArtifactSettings {
            name: name.to_string(),
        };
        let artifact = self
            .experiment_handle
            .load_artifact::<CheckpointRecordSources<B, R>>(name, &settings)
            .map_err(|e| RecorderError::Unknown(format!("Failed to load artifact: {e}")))?;
        Ok(artifact.record)
    }

    fn save_item<I: Serialize>(
        &self,
        _item: I,
        _args: Self::RecordArgs,
    ) -> Result<Self::RecordOutput, RecorderError> {
        Err(RecorderError::Unknown(
            "Saving items directly is not supported by RemoteCheckpointRecorder".to_string(),
        ))
    }

    fn load_item<I>(&self, _args: &mut Self::LoadArgs) -> Result<I, RecorderError>
    where
        I: DeserializeOwned,
    {
        Err(RecorderError::Unknown(
            "Loading items directly is not supported by RemoteCheckpointRecorder".to_string(),
        ))
    }
}
