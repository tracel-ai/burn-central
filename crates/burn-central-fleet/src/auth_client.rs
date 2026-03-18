use std::sync::{Arc, RwLock};

use burn_central_client::{
    ClientError, FleetClient,
    request::TelemetryIngestionEvents,
    response::{
        FleetDeviceAuthTokenResponse, FleetModelDownloadResponse, FleetSyncSnapshotResponse,
    },
};

use crate::{DeviceMetadata, FleetRegistrationToken, state::AuthToken};

#[derive(Debug, thiserror::Error)]
pub enum AuthenticatedFleetClientError {
    #[error("missing auth token for fleet client request")]
    MissingAuthToken,
    #[error("fleet client auth token lock poisoned")]
    PoisonedAuthToken,
    #[error(transparent)]
    Client(#[from] ClientError),
}

#[derive(Debug, Clone)]
pub struct AuthenticatedFleetClient {
    client: FleetClient,
    auth_token: Arc<RwLock<Option<AuthToken>>>,
}

impl AuthenticatedFleetClient {
    pub fn new(client: FleetClient, auth_token: Option<AuthToken>) -> Self {
        Self {
            client,
            auth_token: Arc::new(RwLock::new(auth_token)),
        }
    }

    pub fn is_ready(&self) -> bool {
        self.auth_token
            .read()
            .map(|token| token.as_ref().map(|t| t.is_valid()).unwrap_or(false))
            .unwrap_or(false)
    }

    pub fn set_auth_token(&self, token: AuthToken) -> Result<(), AuthenticatedFleetClientError> {
        let mut auth_token = self
            .auth_token
            .write()
            .map_err(|_| AuthenticatedFleetClientError::PoisonedAuthToken)?;
        *auth_token = Some(token);
        Ok(())
    }

    pub fn clear_auth_token(&self) -> Result<(), AuthenticatedFleetClientError> {
        let mut auth_token = self
            .auth_token
            .write()
            .map_err(|_| AuthenticatedFleetClientError::PoisonedAuthToken)?;
        *auth_token = None;
        Ok(())
    }

    pub fn register(
        &self,
        registration_token: impl Into<FleetRegistrationToken>,
        identity_key: impl Into<String>,
        metadata: Option<DeviceMetadata>,
    ) -> Result<FleetDeviceAuthTokenResponse, ClientError> {
        self.client
            .register(registration_token, identity_key, metadata)
    }

    pub fn sync(
        &self,
        metadata: Option<DeviceMetadata>,
    ) -> Result<FleetSyncSnapshotResponse, AuthenticatedFleetClientError> {
        self.with_token(|token| Ok(self.client.sync(token, metadata)?))
    }

    pub fn model_download(
        &self,
    ) -> Result<FleetModelDownloadResponse, AuthenticatedFleetClientError> {
        self.with_token(|token| Ok(self.client.model_download(token)?))
    }

    pub fn ingest_telemetry(
        &self,
        events: TelemetryIngestionEvents,
    ) -> Result<(), AuthenticatedFleetClientError> {
        self.with_token(|token| Ok(self.client.ingest_telemetry(token, events)?))
    }

    fn with_token<Ret>(
        &self,
        f: impl FnOnce(&str) -> Result<Ret, AuthenticatedFleetClientError>,
    ) -> Result<Ret, AuthenticatedFleetClientError> {
        self.auth_token
            .read()
            .map_err(|_| AuthenticatedFleetClientError::PoisonedAuthToken)?
            .as_ref()
            .map(|token| token.token())
            .ok_or(AuthenticatedFleetClientError::MissingAuthToken)
            .and_then(|token| f(token))
    }
}
