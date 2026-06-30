#![recursion_limit = "256"]

use burn::backend::wgpu::WgpuDevice;
use burn::tensor::Device;
use mnist::training::{self, MnistTrainingConfig};

use tracel::experiment::ExperimentRun;
use tracel::{Connection, Context};

fn main() -> anyhow::Result<()> {
    Context::new(Connection::Cloud)?
        .experiment()
        .create("mnist_wgpu", |session: &ExperimentRun, config| {
            training::run(
                session,
                config,
                vec![Device::autodiff(WgpuDevice::default().into())],
            )
        })
        .run(MnistTrainingConfig::default())
        .map_err(|e| anyhow::anyhow!("training failed: {e}"))?;

    Ok(())
}
