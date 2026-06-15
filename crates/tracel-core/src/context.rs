use std::sync::Arc;

use crate::connection::{Connection, ContextError};
use tracel_experiment::ExperimentModule;
use tracel_experiment::ExperimentProvider;

#[derive(Clone)]
pub struct Context {
    experiment_provider: Arc<dyn ExperimentProvider>,
}

impl Context {
    pub fn new(connexion: Connection) -> Result<Self, ContextError> {
        let providers = connexion.into_providers()?;
        Ok(Self {
            experiment_provider: providers.experiment,
        })
    }

    pub fn experiment(&self) -> ExperimentModule {
        ExperimentModule::new(self.experiment_provider.clone())
    }
}
