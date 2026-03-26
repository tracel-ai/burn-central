//! Error types returned by experiment operations.

/// Broad category for an [`ExperimentError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentErrorKind {
    /// The operation stopped because cancellation was observed.
    Cancelled,

    /// The operation attempted to use a run that has already completed.
    AlreadyFinished,

    /// The handle points to a run that is no longer active.
    InactiveRun,

    /// Artifact encoding, decoding, or transport failed.
    Artifact,

    /// Internal or backend-specific failure that does not fit another category.
    Internal,
}

/// Error returned by experiment operations.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ExperimentError {
    /// High-level category for the failure.
    pub kind: ExperimentErrorKind,

    /// Human-readable error summary.
    pub message: String,

    /// Optional lower-level source error.
    #[source]
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl ExperimentError {
    pub(crate) fn new(kind: ExperimentErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn with_source<E>(
        kind: ExperimentErrorKind,
        message: impl Into<String>,
        source: E,
    ) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
    {
        Self {
            kind,
            message: message.into(),
            source: Some(source.into()),
        }
    }
}
