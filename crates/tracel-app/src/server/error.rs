use std::error::Error;

use crate::job_register::JobRegisterError;

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("server error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("unknown job '{name}'. Available: {}", available.join(", "))]
    UnknownJob {
        name: String,
        available: Vec<String>,
    },

    #[error("validation failed: {0}")]
    ValidationFailed(#[source] Box<dyn Error + Send + Sync>),

    #[error("execution failed: {0}")]
    ExecutionFailed(#[source] Box<dyn Error + Send + Sync>),
}

impl From<JobRegisterError> for ServerError {
    fn from(err: JobRegisterError) -> Self {
        match err {
            JobRegisterError::UnknownJob { name, available } => {
                ServerError::UnknownJob { name, available }
            }
            JobRegisterError::ValidationFailed(e) => ServerError::ValidationFailed(e),
            JobRegisterError::ExecutionFailed(e) => ServerError::ExecutionFailed(e),
        }
    }
}
