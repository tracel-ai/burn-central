use crate::{model, state};

#[derive(Debug, thiserror::Error)]
pub enum FleetError {
    #[error("fleet sync failed: {0}")]
    SyncFailed(String),
    #[error("fleet model download failed: {0}")]
    DownloadFailed(String),
    #[error("failed to determine cache directory")]
    CacheDirUnavailable,
    #[error(transparent)]
    State(#[from] state::FleetStateStoreError),
    #[error(transparent)]
    Model(#[from] model::ModelCacheError),
}
