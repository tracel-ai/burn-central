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
