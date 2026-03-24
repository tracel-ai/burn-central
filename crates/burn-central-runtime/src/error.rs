use burn_central_experiment::error::ExperimentError;

#[derive(thiserror::Error, Debug)]
pub enum RuntimeError {
    #[error("Handler '{0}' not found")]
    HandlerNotFound(String),
    #[error("Client initialization failed: {0}")]
    ClientInitializationFailed(String),
    #[error("Experiment API call failed: {0}")]
    ExperimentApiFailed(#[from] ExperimentError),
    #[error("Handler execution failed: {0}")]
    HandlerFailed(anyhow::Error),
    #[error("Ambiguous target '{0}'. Found multiple handlers: {1:?}")]
    AmbiguousHandlerName(String, Vec<String>),
    #[error("Invalid arguments: {0}")]
    InvalidArgs(String),
}
