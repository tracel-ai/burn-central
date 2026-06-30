#![recursion_limit = "256"]

use burn::backend::{FlexDevice, wgpu::WgpuDevice};
use burn::tensor::Device;
use mnist::training::{self, MnistTrainingConfig};

use tracel::app::server::Server;
use tracel::experiment::ExperimentRun;
use tracel::{Connection, Context};

fn main() -> anyhow::Result<()> {
    let module = Context::new(Connection::Cloud)?.experiment();
    let job = module.create("mnist_wgpu", |session: &ExperimentRun, config| {
        training::run(
            session,
            config,
            vec![Device::autodiff(WgpuDevice::default().into())],
        )
    });
    let default_job = module.create("mnist_flex", |session: &ExperimentRun, config| {
        training::run(
            session,
            config,
            vec![Device::autodiff(FlexDevice::default().into())],
        )
    });

    let no_default_job = module.create("mnist_no_default", |session: &ExperimentRun, config| {
        training::run(
            session,
            config,
            vec![Device::autodiff(WgpuDevice::default().into())],
        )
    });

    Server::new()
        .port(3000)
        .register_with_default(job, MnistTrainingConfig::default())
        .register_with_default(default_job, MnistTrainingConfig::small())
        .register(no_default_job)
        .run()?;

    Ok(())
}
