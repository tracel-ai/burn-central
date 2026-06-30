use crate::{
    job::{Job, RunFn, ValidateFn},
    mapper::Mapper,
};
use std::{any::Any, collections::HashMap, error::Error};

#[derive(Debug, thiserror::Error)]
pub(crate) enum JobRegisterError {
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

struct JobEntry {
    validate: ValidateFn,
    run: RunFn,
}

#[derive(Default)]
pub struct JobRegister {
    jobs: HashMap<String, JobEntry>,
}

impl JobRegister {
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
        }
    }

    fn erase_job<J, I, O, M>(job: J, mapper: M) -> JobEntry
    where
        J: Job<I, O> + Send + Sync + 'static,
        M: Mapper<I> + Send + Sync + 'static,
        I: Send + 'static,
        O: 'static,
    {
        let validate: ValidateFn = Box::new(move |config_str: &str| {
            let input = mapper
                .map(config_str)
                .map_err(JobRegisterError::ValidationFailed)?;
            Ok(Box::new(input) as Box<dyn Any + Send>)
        });

        let run: RunFn = Box::new(move |input: Box<dyn Any + Send>| {
            let input = *input.downcast::<I>().map_err(|_| {
                JobRegisterError::ExecutionFailed("internal type mismatch in job dispatch".into())
            })?;
            job.execute(input)
                .map(|_| ())
                .map_err(JobRegisterError::ExecutionFailed)
        });

        JobEntry { validate, run }
    }

    pub fn register<J, I, O, F>(mut self, job: J, mapper: F) -> Self
    where
        J: Job<I, O> + Send + Sync + 'static,
        F: Mapper<I> + Send + Sync + 'static,
        I: Send + 'static,
        O: 'static,
    {
        let name = job.name().to_string();
        if self.jobs.contains_key(&name) {
            panic!("job '{}' is already registered", name);
        }
        let entry = Self::erase_job(job, mapper);
        self.jobs.insert(name, entry);
        self
    }

    fn job_names(&self) -> Vec<String> {
        self.jobs.keys().cloned().collect()
    }

    pub fn validate(
        &self,
        job_name: &str,
        config: &str,
    ) -> Result<Box<dyn Any + Send>, JobRegisterError> {
        let entry = self
            .jobs
            .get(job_name)
            .ok_or_else(|| JobRegisterError::UnknownJob {
                name: job_name.to_string(),
                available: self.job_names(),
            })?;
        (entry.validate)(config)
    }

    pub fn run(&self, job_name: &str, input: Box<dyn Any + Send>) -> Result<(), JobRegisterError> {
        let entry = self
            .jobs
            .get(job_name)
            .ok_or_else(|| JobRegisterError::UnknownJob {
                name: job_name.to_string(),
                available: self.job_names(),
            })?;
        (entry.run)(input)
    }

    pub fn dispatch(&self, job_name: &str, config: &str) -> Result<(), JobRegisterError> {
        let input = self.validate(job_name, config)?;
        self.run(job_name, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::Job;
    use crate::mapper::Mapper;
    use std::error::Error;

    struct FakeJob {
        name: &'static str,
        should_fail: bool,
    }

    impl Job<String, ()> for FakeJob {
        fn name(&self) -> &str {
            self.name
        }

        fn execute(&self, _input: String) -> Result<(), Box<dyn Error + Send + Sync>> {
            if self.should_fail {
                Err("job execution failed".into())
            } else {
                Ok(())
            }
        }
    }

    struct FakeMapper {
        should_fail: bool,
    }

    impl Mapper<String> for FakeMapper {
        fn map(&self, raw: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            if self.should_fail {
                Err("mapper failed".into())
            } else {
                Ok(raw.to_string())
            }
        }
    }

    #[test]
    fn given_unknown_job_when_validating_then_return_unknown_job_error() {
        let register = JobRegister::new().register(
            FakeJob {
                name: "train",
                should_fail: false,
            },
            FakeMapper { should_fail: false },
        );

        let result = register.validate("infer", "{}");

        assert!(matches!(result, Err(JobRegisterError::UnknownJob { .. })));
    }

    #[test]
    fn given_bad_config_when_validating_then_return_validation_failed_error() {
        let register = JobRegister::new().register(
            FakeJob {
                name: "train",
                should_fail: false,
            },
            FakeMapper { should_fail: true },
        );

        let result = register.validate("train", "{}");

        assert!(matches!(result, Err(JobRegisterError::ValidationFailed(_))));
    }

    #[test]
    fn given_good_input_when_validating_then_return_input() {
        let register = JobRegister::new().register(
            FakeJob {
                name: "train",
                should_fail: false,
            },
            FakeMapper { should_fail: false },
        );

        let result = register.validate("train", "hello");

        assert!(result.is_ok());
    }

    #[test]
    fn given_unknown_job_when_running_then_return_unknown_job_error() {
        let register = JobRegister::new().register(
            FakeJob {
                name: "train",
                should_fail: false,
            },
            FakeMapper { should_fail: false },
        );

        let input: Box<dyn Any + Send> = Box::new("test".to_string());
        let result = register.run("infer", input);

        assert!(matches!(result, Err(JobRegisterError::UnknownJob { .. })));
    }

    #[test]
    fn given_job_error_when_running_then_return_execution_failed_error() {
        let register = JobRegister::new().register(
            FakeJob {
                name: "train",
                should_fail: true,
            },
            FakeMapper { should_fail: false },
        );

        let input = register.validate("train", "{}").unwrap();
        let result = register.run("train", input);

        assert!(matches!(result, Err(JobRegisterError::ExecutionFailed(_))));
    }

    #[test]
    fn given_correct_inputs_when_dispatching_then_return_ok() {
        let register = JobRegister::new().register(
            FakeJob {
                name: "train",
                should_fail: false,
            },
            FakeMapper { should_fail: false },
        );

        let result = register.dispatch("train", "hello");

        assert!(result.is_ok());
    }

    #[test]
    fn given_unknown_job_when_dispatching_then_return_unknown_job_error() {
        let register = JobRegister::new().register(
            FakeJob {
                name: "train",
                should_fail: false,
            },
            FakeMapper { should_fail: false },
        );

        let result = register.dispatch("infer", "{}");

        assert!(matches!(result, Err(JobRegisterError::UnknownJob { .. })));
    }

    #[test]
    fn given_failing_mapper_when_dispatching_then_return_validation_failed() {
        let register = JobRegister::new().register(
            FakeJob {
                name: "train",
                should_fail: false,
            },
            FakeMapper { should_fail: true },
        );

        let result = register.dispatch("train", "{}");

        assert!(matches!(result, Err(JobRegisterError::ValidationFailed(_))));
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn given_duplicate_job_name_when_registering_then_panic() {
        JobRegister::new()
            .register(
                FakeJob {
                    name: "train",
                    should_fail: false,
                },
                FakeMapper { should_fail: false },
            )
            .register(
                FakeJob {
                    name: "train",
                    should_fail: false,
                },
                FakeMapper { should_fail: false },
            );
    }

    #[test]
    fn given_mismatched_input_type_when_running_then_return_execution_failed_error() {
        let register = JobRegister::new().register(
            FakeJob {
                name: "train",
                should_fail: false,
            },
            FakeMapper { should_fail: false },
        );

        let input: Box<dyn Any + Send> = Box::new(42i32);
        let result = register.run("train", input);

        assert!(matches!(result, Err(JobRegisterError::ExecutionFailed(_))));
    }

    #[test]
    fn given_failing_job_when_dispatching_then_return_execution_failed() {
        let register = JobRegister::new().register(
            FakeJob {
                name: "train",
                should_fail: true,
            },
            FakeMapper { should_fail: false },
        );

        let result = register.dispatch("train", "{}");

        assert!(matches!(result, Err(JobRegisterError::ExecutionFailed(_))));
    }
}
