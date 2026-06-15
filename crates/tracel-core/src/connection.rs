use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "station")]
use url::Url;

use crate::backend::cloud::{CloudBackend, CloudError};
use crate::backend::local::LocalBackend;
#[cfg(feature = "station")]
use crate::backend::station::StationBackend;
use tracel_experiment::ExperimentProvider;

pub struct Providers {
    pub experiment: Arc<dyn ExperimentProvider>,
}

#[derive(Debug, Clone)]
pub enum Connection {
    Cloud,
    Offline(PathBuf),
    #[cfg(feature = "station")]
    Station(Url),
}

impl Connection {
    pub(crate) fn into_providers(self) -> Result<Providers, ContextError> {
        match self {
            Connection::Cloud => {
                let backend = Arc::new(CloudBackend::create_context()?);
                Ok(Providers {
                    experiment: backend,
                })
            }
            Connection::Offline(path) => {
                let backend = Arc::new(LocalBackend::create_context(path));
                Ok(Providers {
                    experiment: backend,
                })
            }
            #[cfg(feature = "station")]
            Connection::Station(url) => {
                let backend = Arc::new(StationBackend::create_context(url));
                Ok(Providers {
                    experiment: backend,
                })
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error(transparent)]
    Cloud(#[from] CloudError),
}
