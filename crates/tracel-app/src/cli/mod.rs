mod error;

pub use error::CliError;

use crate::{job::Job, job_register::JobRegister, mapper::Mapper};
use clap::Parser;
use std::error::Error;

#[derive(Parser)]
#[command(about = "Run a registered job")]
struct Args {
    job: Option<String>,
    config: Option<String>,
}

struct DefaultJob {
    runner: Box<dyn FnOnce() -> Result<(), Box<dyn Error + Send + Sync>>>,
}

#[derive(Default)]
pub struct Cli {
    register: JobRegister,
    default_job: Option<DefaultJob>,
}

impl Cli {
    pub fn new() -> Self {
        Self {
            register: JobRegister::new(),
            default_job: None,
        }
    }

    pub fn register<J, I, O, F>(mut self, job: J, mapper: F) -> Self
    where
        J: Job<I, O> + Send + Sync + 'static,
        F: Mapper<I> + Send + Sync + 'static,
        I: Send + 'static,
        O: 'static,
    {
        self.register = self.register.register(job, mapper);
        self
    }

    pub fn default_job<J, I, O>(mut self, job: J, config: I) -> Self
    where
        J: Job<I, O> + Send + Sync + 'static,
        I: Send + 'static,
        O: 'static,
    {
        self.default_job = Some(DefaultJob {
            runner: Box::new(move || job.execute(config).map(|_| ())),
        });
        self
    }

    pub fn run(self) -> Result<(), CliError> {
        let args = Args::parse();
        self.dispatch(args.job, args.config)
    }

    fn dispatch(self, job: Option<String>, config: Option<String>) -> Result<(), CliError> {
        match job {
            Some(job_name) => {
                let config_str = config.unwrap_or_default();
                self.register.dispatch(&job_name, &config_str)?;
                Ok(())
            }
            None => {
                let d = self.default_job.ok_or(CliError::MissingDefault)?;
                (d.runner)().map_err(CliError::ExecutionFailed)
            }
        }
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

    impl FakeJob {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                should_fail: false,
            }
        }

        fn failing(name: &'static str) -> Self {
            Self {
                name,
                should_fail: true,
            }
        }
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

    impl FakeMapper {
        fn new() -> Self {
            Self { should_fail: false }
        }

        fn failing() -> Self {
            Self { should_fail: true }
        }
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
    fn given_registered_job_when_dispatching_named_job_then_return_ok() {
        let cli = Cli::new().register(FakeJob::new("train"), FakeMapper::new());

        let result = cli.dispatch(Some("train".into()), Some("{}".into()));

        assert!(result.is_ok());
    }

    #[test]
    fn given_unknown_job_name_when_dispatching_then_return_unknown_job_error() {
        let cli = Cli::new().register(FakeJob::new("train"), FakeMapper::new());

        let result = cli.dispatch(Some("infer".into()), Some("{}".into()));

        assert!(matches!(result, Err(CliError::UnknownJob { .. })));
    }

    #[test]
    fn given_no_config_when_dispatching_named_job_then_default_config_to_empty_string() {
        let cli = Cli::new().register(FakeJob::new("train"), FakeMapper::new());

        let result = cli.dispatch(Some("train".into()), None);

        assert!(result.is_ok());
    }

    #[test]
    fn given_mapper_error_when_dispatching_then_return_validation_failed_error() {
        let cli = Cli::new().register(FakeJob::new("train"), FakeMapper::failing());

        let result = cli.dispatch(Some("train".into()), Some("{}".into()));

        assert!(matches!(result, Err(CliError::ValidationFailed(_))));
    }

    #[test]
    fn given_job_error_when_dispatching_then_return_execution_failed_error() {
        let cli = Cli::new().register(FakeJob::failing("train"), FakeMapper::new());

        let result = cli.dispatch(Some("train".into()), Some("{}".into()));

        assert!(matches!(result, Err(CliError::ExecutionFailed(_))));
    }

    #[test]
    fn given_default_job_when_dispatching_with_no_job_name_then_return_ok() {
        let cli = Cli::new().default_job(FakeJob::new("default"), "config".to_string());

        let result = cli.dispatch(None, None);

        assert!(result.is_ok());
    }

    #[test]
    fn given_no_job_and_no_default_when_dispatching_then_return_missing_default_error() {
        let cli = Cli::new();

        let result = cli.dispatch(None, None);

        assert!(matches!(result, Err(CliError::MissingDefault)));
    }

    #[test]
    fn given_failing_default_job_when_dispatching_then_return_execution_failed() {
        let cli = Cli::new().default_job(FakeJob::failing("default"), "config".to_string());

        let result = cli.dispatch(None, None);

        assert!(matches!(result, Err(CliError::ExecutionFailed(_))));
    }

    #[test]
    fn given_default_job_and_named_job_when_dispatching_named_job_then_run_registered_job_not_default()
     {
        let cli = Cli::new()
            .register(FakeJob::new("train"), FakeMapper::new())
            .default_job(FakeJob::failing("default"), "config".to_string());

        let result = cli.dispatch(Some("train".into()), Some("{}".into()));

        assert!(result.is_ok());
    }

    #[test]
    fn given_multiple_registered_jobs_when_dispatching_by_name_then_run_correct_job() {
        let cli = Cli::new()
            .register(FakeJob::new("train"), FakeMapper::new())
            .register(FakeJob::new("infer"), FakeMapper::new());

        let result = cli.dispatch(Some("infer".into()), Some("{}".into()));

        assert!(result.is_ok());
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn given_duplicate_job_name_when_registering_then_panic() {
        Cli::new()
            .register(FakeJob::new("train"), FakeMapper::new())
            .register(FakeJob::new("train"), FakeMapper::new());
    }
}
