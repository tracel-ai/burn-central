use burn_central_client::Client;

use crate::{
    CancelToken, ExperimentId, ExperimentRun,
    remote::base::{BurnCentralArtifactReader, BurnCentralError, BurnCentralSession},
};

mod artifacts;
mod base;
mod log_store;
mod socket;

#[derive(Debug, Clone)]
pub struct ExperimentPath {
    owner_name: String,
    project_name: String,
    experiment_num: i32,
}

impl ExperimentPath {
    pub fn new(
        owner_name: impl Into<String>,
        project_name: impl Into<String>,
        experiment_num: i32,
    ) -> Self {
        Self {
            owner_name: owner_name.into(),
            project_name: project_name.into(),
            experiment_num,
        }
    }

    pub fn owner_name(&self) -> &str {
        &self.owner_name
    }

    pub fn project_name(&self) -> &str {
        &self.project_name
    }

    pub fn experiment_num(&self) -> i32 {
        self.experiment_num
    }
}

pub struct RemoteExperimentId(i32);

impl RemoteExperimentId {
    pub fn new(num: i32) -> Self {
        Self(num)
    }

    pub fn to_experiment_id(&self) -> ExperimentId {
        ExperimentId::from(format!("{}", self.0))
    }

    pub fn from_experiment_id(id: &ExperimentId) -> Option<Self> {
        id.parse().map(RemoteExperimentId)
    }

    pub fn num(&self) -> i32 {
        self.0
    }
}

pub fn create_experiment_run(
    client: Client,
    namespace: &str,
    project_name: &str,
    digest: String,
    routine: String,
) -> Result<ExperimentRun, BurnCentralError> {
    let experiment = client
        .create_experiment(namespace, project_name, None, digest, routine)
        .map_err(|e| BurnCentralError::Client {
            context: format!("Failed to create experiment for {namespace}/{project_name}"),
            source: e,
        })?;

    let experiment_num = experiment.experiment_num;
    let path = ExperimentPath::new(namespace, project_name, experiment_num);
    let cancel_token = CancelToken::new();
    let session = BurnCentralSession::new(client.clone(), path.clone(), cancel_token.clone())?;
    let reader = BurnCentralArtifactReader::new(client, path);

    let id = RemoteExperimentId::new(experiment_num).to_experiment_id();

    Ok(ExperimentRun::new(id, session, reader, cancel_token))
}
