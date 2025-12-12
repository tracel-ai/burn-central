use std::str::FromStr;

use crate::client::BurnCentralError;
use once_cell::sync::Lazy;
use regex::Regex;

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectPath {
    pub owner_name: String,
    pub project_name: String,
}

impl ProjectPath {
    pub fn new(owner_name: impl Into<String>, project_name: impl Into<String>) -> Self {
        ProjectPath {
            owner_name: owner_name.into(),
            project_name: project_name.into(),
        }
    }

    pub fn validate_path(path: &str) -> bool {
        static NAME_REGEX: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"^[a-zA-Z0-9_.-]+$")
                .expect("Should be able to compile name validation regex.")
        });

        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() != 2 {
            return false;
        }

        for part in parts {
            if !NAME_REGEX.is_match(part) {
                return false;
            }
        }

        true
    }

    pub fn owner_name(&self) -> &str {
        &self.owner_name
    }

    pub fn project_name(&self) -> &str {
        &self.project_name
    }
}

impl TryFrom<String> for ProjectPath {
    type Error = BurnCentralError;

    fn try_from(path: String) -> Result<Self, Self::Error> {
        if !ProjectPath::validate_path(&path) {
            return Err(Self::Error::InvalidProjectPath(path));
        }

        let parts: Vec<&str> = path.split('/').collect();
        Ok(ProjectPath {
            owner_name: parts[0].into(),
            project_name: parts[1].into(),
        })
    }
}

impl std::fmt::Display for ProjectPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.owner_name, self.project_name)
    }
}

impl FromStr for ProjectPath {
    type Err = BurnCentralError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ProjectPath::try_from(s.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct ExperimentPath {
    project_path: ProjectPath,
    experiment_num: i32,
}

impl ExperimentPath {
    pub fn new(
        owner_name: impl Into<String>,
        project_name: impl Into<String>,
        experiment_num: i32,
    ) -> Self {
        Self {
            project_path: ProjectPath::new(owner_name, project_name),
            experiment_num,
        }
    }

    pub fn validate_path(path: &str) -> bool {
        static NAME_REGEX: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"^[a-zA-Z0-9_.-]+$")
                .expect("Should be able to compile name validation regex.")
        });

        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() != 3 {
            return false;
        }

        if !NAME_REGEX.is_match(parts[0]) || !NAME_REGEX.is_match(parts[1]) {
            return false;
        }

        if parts[2].parse::<i32>().is_err() {
            return false;
        }

        true
    }

    pub fn owner_name(&self) -> &str {
        &self.project_path.owner_name
    }

    pub fn project_name(&self) -> &str {
        &self.project_path.project_name
    }

    pub fn experiment_num(&self) -> i32 {
        self.experiment_num
    }
}

impl TryFrom<String> for ExperimentPath {
    type Error = BurnCentralError;

    fn try_from(path: String) -> Result<Self, Self::Error> {
        if !ExperimentPath::validate_path(&path) {
            return Err(Self::Error::InvalidExperimentPath(path));
        }

        let parts: Vec<&str> = path.split('/').collect();
        let project_path = ProjectPath::try_from(parts[0..2].join("/"))?;
        let experiment_num = parts[2]
            .parse::<i32>()
            .map_err(|_| Self::Error::InvalidExperimentNumber(parts[2].to_string()))?;

        Ok(ExperimentPath {
            project_path,
            experiment_num,
        })
    }
}

impl std::fmt::Display for ExperimentPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.project_path, self.experiment_num)
    }
}

impl FromStr for ExperimentPath {
    type Err = BurnCentralError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ExperimentPath::try_from(s.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelPath {
    project_path: ProjectPath,
    model_name: String,
}

impl ModelPath {
    pub fn new(namespace: &str, project_name: &str, model_name: &str) -> Self {
        ModelPath {
            project_path: ProjectPath::new(namespace.to_string(), project_name.to_string()),
            model_name: model_name.to_string(),
        }
    }

    pub fn validate_path(path: &str) -> bool {
        static NAME_REGEX: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"^[a-zA-Z0-9_.-]+$")
                .expect("Should be able to compile name validation regex.")
        });

        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() != 3 {
            return false;
        }

        for part in parts {
            if !NAME_REGEX.is_match(part) {
                return false;
            }
        }

        true
    }

    pub fn namespace(&self) -> &str {
        &self.project_path.owner_name
    }

    pub fn project_name(&self) -> &str {
        &self.project_path.project_name
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl TryFrom<String> for ModelPath {
    type Error = BurnCentralError;

    fn try_from(path: String) -> Result<Self, Self::Error> {
        if !ModelPath::validate_path(&path) {
            return Err(Self::Error::InvalidModelPath(path));
        }

        let parts: Vec<&str> = path.split('/').collect();
        let project_path = ProjectPath::try_from(parts[0..2].join("/"))?;
        let model_name = parts[2].to_string();

        Ok(ModelPath {
            project_path,
            model_name,
        })
    }
}

impl FromStr for ModelPath {
    type Err = BurnCentralError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ModelPath::try_from(s.to_string())
    }
}

impl std::fmt::Display for ModelPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.project_path, self.model_name)
    }
}
