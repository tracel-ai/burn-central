#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentErrorKind {
    /// Caused by an attempt to operate on an experiment that has already been cancelled.
    Cancelled,
    /// Caused by an attempt to operate on an experiment that has already been finished (either successfully or with failure).
    AlreadyFinished,
    /// Caused by an attempt to use an experiment handle that points to an experiment that is no longer active (e.g., due to cancellation or completion).
    InactiveRun,
    /// Caused by an error during artifact related operations.
    Artifact,
    /// Caused by an internal error. This is a catch-all for errors that don't fit into the other categories.
    Internal,
}

/// The main error type for experiment operations.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ExperimentError {
    pub kind: ExperimentErrorKind,
    pub message: String,
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
