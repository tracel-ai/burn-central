//! This module provides the [BurnCentral] struct, which is used to interact with the Burn Central service.

use crate::artifacts::ExperimentArtifactClient;
use crate::experiment::{ExperimentRun, ExperimentTrackerError};
use crate::models::ModelRegistry;
use crate::schemas::ExperimentPath;
use burn_central_client::{BurnCentralCredentials, Client, ClientError};
use reqwest::Url;

/// Errors that can occur during the initialization of the [BurnCentral] client.
#[derive(Debug, thiserror::Error)]
pub enum InitError {
    /// Represents an error related to the client.
    #[error("Client error: {0}")]
    Client(#[from] ClientError),
    /// Represents an error when the endpoint URL is invalid.
    #[error("Failed to parse endpoint URL: {0}")]
    InvalidEndpointUrl(String),
}

#[derive(Debug, thiserror::Error)]
/// Errors that can occur during when using the [BurnCentral] client.
///
/// Those errors should be handled by the user of this library. If he want to implement fallback behavior or retry logic.
pub enum BurnCentralError {
    /// Input validation errors
    #[error("Invalid experiment path: {0}")]
    InvalidExperimentPath(String),
    /// This error occurs when the provided project path is invalid. It is a input validation
    /// errors. No API call has been made yet.
    #[error("Invalid project path: {0}")]
    InvalidProjectPath(String),
    /// This error occurs when the provided experiment number is invalid. It is a input validation
    /// errors. No API call has been made yet.
    #[error("Invalid experiment number: {0}")]
    InvalidExperimentNumber(String),
    /// This error occurs when the provided model path is invalid. It is a input validation
    /// errors. No API call has been made yet.
    #[error("Invalid model path: {0}")]
    InvalidModelPath(String),

    /// Represents an error related to client operations.
    ///
    /// This error variant is used to encapsulate client-specific errors along with additional context
    /// and the underlying source error for more detailed debugging.
    ///
    /// # Fields
    /// - `context` (String): A description or additional information about the client error context.
    /// - `source` (ClientError): The underlying source of the client error, providing more details about the cause.
    #[error("Client error: {context}\nSource: {source}")]
    Client {
        context: String,
        source: ClientError,
    },
    /// Represents an error related to the experiment tracker. Those errors are coming from the
    /// websocket connection that open when starting an experiment run.
    #[error("Experiment error: {0}")]
    ExperimentTracker(#[from] ExperimentTrackerError),

    /// Error that should be used when the user is not logged in but tries to perform an operation that requires authentication.
    #[error("The user is not authenticated.")]
    Unauthenticated,

    /// Error that should be used when the client performs operations that can fail due to IO issues.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Error that should be used when the client encounters an error that is not specifically handled.
    #[error("Internal error: {0}")]
    Internal(String),
}

/// This builder struct is used to create a [BurnCentral] client. It should only be used by the
/// burn_central generated crates.
#[doc(hidden)]
pub struct BurnCentralBuilder {
    endpoint: Option<String>,
    credentials: BurnCentralCredentials,
}

impl BurnCentralBuilder {
    /// Creates a new [BurnCentralBuilder] with the given credentials.
    pub fn new(credentials: impl Into<BurnCentralCredentials>) -> Self {
        BurnCentralBuilder {
            endpoint: None,
            credentials: credentials.into(),
        }
    }

    /// Sets the endpoint for the [BurnCentral] client.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Builds the [BurnCentral] client.
    pub fn build(self) -> Result<BurnCentral, InitError> {
        let url = match self.endpoint {
            Some(s) => s
                .parse::<Url>()
                .map_err(|e| InitError::InvalidEndpointUrl(e.to_string()))?,
            None => {
                Url::parse("https://central.burn.dev/api/").expect("Default URL should be valid")
            }
        };
        #[allow(deprecated)]
        let client = Client::from_url(url, &self.credentials)?;
        Ok(BurnCentral::new(client))
    }
}

/// This struct provides the main interface to interact with Burn Central.
///
/// It wrap the burn_central_client [Client] and provides higher level methods to interact with
/// experiments, models, and artifacts.
#[derive(Clone)]
pub struct BurnCentral {
    client: Client,
}

impl BurnCentral {
    /// Creates a new [BurnCentral] instance with the given credentials.
    ///
    /// This is to be used by the burn central generated crates. It is internal tooling and user
    /// should not seek to connect with it directly as they would expose their credentials in the
    /// code.
    #[doc(hidden)]
    pub fn login(credentials: impl Into<BurnCentralCredentials>) -> Result<Self, InitError> {
        let credentials = credentials.into();
        BurnCentralBuilder::new(credentials).build()
    }

    /// Creates a new [BurnCentralBuilder] to configure the client.
    ///
    /// This is to be used by the burn central generated crates. It is internal tooling and user
    /// should not seek to connect with it directly as they would expose their credentials in the
    /// code.
    #[doc(hidden)]
    pub fn builder(credentials: impl Into<BurnCentralCredentials>) -> BurnCentralBuilder {
        BurnCentralBuilder::new(credentials)
    }

    /// Creates a new instance of [BurnCentral] with the given [Client].
    fn new(client: Client) -> Self {
        BurnCentral { client }
    }

    /// Start a new experiment. This will create a new experiment on the Burn Central backend and start it.
    pub fn start_experiment(
        &self,
        namespace: &str,
        project_name: &str,
        digest: String,
        routine: String,
    ) -> Result<ExperimentRun, BurnCentralError> {
        let experiment = self
            .client
            .create_experiment(namespace, project_name, None, digest, routine)
            .map_err(|e| BurnCentralError::Client {
                context: format!("Failed to create experiment for {namespace}/{project_name}"),
                source: e,
            })?;
        println!("Experiment num: {}", experiment.experiment_num);

        ExperimentRun::new(
            self.client.clone(),
            namespace,
            project_name,
            experiment.experiment_num,
        )
        .map_err(BurnCentralError::ExperimentTracker)
    }

    pub fn artifacts(
        &self,
        owner: &str,
        project: &str,
        exp_num: i32,
    ) -> Result<ExperimentArtifactClient, BurnCentralError> {
        let exp_path = ExperimentPath::new(owner, project, exp_num);
        Ok(ExperimentArtifactClient::new(self.client.clone(), exp_path))
    }

    /// Create a model registry for downloading models from Burn Central.
    /// Models are project-scoped and identified by namespace/project/model_name.
    pub fn models(&self) -> ModelRegistry {
        ModelRegistry::new(self.client.clone())
    }
}
