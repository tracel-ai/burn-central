/// Command-line interface for dispatching registered jobs.
pub mod cli;
/// Configuration mappers for deserializing job inputs. (Mostly use for the cli)
pub mod mapper;
/// HTTP server for dispatching registered jobs via endpoints.
#[cfg(feature = "server")]
pub mod server;

mod job;
mod job_register;
