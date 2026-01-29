<div align="center">

<h1>Burn central</h1>

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-central)](https://crates.io/crates/burn-central)
[![Minimum Supported Rust Version](https://img.shields.io/crates/msrv/burn-central)](https://crates.io/crates/burn-central)
[![Test Status](https://github.com/tracel-ai/burn-central/actions/workflows/ci.yml/badge.svg)](https://github.com/tracel-ai/burn-central/actions/workflows/ci.yml)
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)

---
</div>


## Description

Burn Central is a new way of using Burn. It aims at providing a central platform for experiment tracking, model sharing, and deployment for all Burn users!

This repository contains the SDK associated with the project. It offers macros that help attach to your code and send training data to our application. To use this project you must first create an account on the [application](https://s1-central.burn.dev/).

Also needed to use this is the new [burn-cli](https://github.com/tracel-ai/burn-central-cli).

## Installation

Add Burn Central to your `Cargo.toml`:

```toml
[dependencies]
burn-central = "0.1.0"
```

## Quick Start

Currently, we only support training. Here's how to integrate Burn Central into your training workflow:

### 1. Register your training function

Use the `#[register]` macro to register your training function:

```rust
use burn_central::{
    experiment::ExperimentRun,
    macros::register,
    runtime::{Args, ArtifactLoader, Model, MultiDevice},
};
use burn::prelude::*;

#[register(training, name = "mnist")]
pub fn training<B: AutodiffBackend>(
    client: &ExperimentRun,
    config: Args<YourExperimentConfig>,
    MultiDevice(devices): MultiDevice<B>,
    loader: ArtifactLoader<ModelArtifact<B>>,
) -> Result<Model<ModelArtifact<B::InnerBackend>>, String> {
    // Log your configuration
    client.log_config("Training Config", &training_config)
        .expect("Logging config failed");

    // Your training logic here...
    let model = train::<B>(client, artifact_dir, &training_config, devices[0].clone())?;

    Ok(Model(ModelArtifact {
        model_record: model.into_record(),
        config: training_config,
    }))
}
```

### 2. Integrate with your Learner

To enable experiment tracking, you need to add three key components to your `LearnerBuilder`:

```rust
use burn_central::{
    log::RemoteExperimentLoggerInstaller,
    metrics::RemoteMetricLogger,
    record::RemoteCheckpointRecorder,
};
use burn::train::{LearnerBuilder, metric::{AccuracyMetric, LossMetric}};

let learner = LearnerBuilder::new(artifact_dir)
    .metric_train_numeric(AccuracyMetric::new())
    .metric_valid_numeric(AccuracyMetric::new())
    .metric_train_numeric(LossMetric::new())
    .metric_valid_numeric(LossMetric::new())
    // Required: Remote metric logging
    .with_metric_logger(RemoteMetricLogger::new(client))
    // Required: Remote checkpoint saving
    .with_file_checkpointer(RemoteCheckpointRecorder::new(client))
    .num_epochs(config.num_epochs)
    .summary()
    .build(
        model.init::<B>(&device),
        optimizer.init(),
        learning_rate,
        LearningStrategy::SingleDevice(device),
    );
```

### 3. Run your training

Once integrated, run your training using the [burn-cli](https://github.com/tracel-ai/burn-central-cli) to automatically track metrics, checkpoints, and logs on Burn Central.

## Requirements

- Rust 1.87.0 or higher
- A Burn Central account (create one at [central.burn.dev](https://central.burn.dev/))
- The [burn-cli](https://github.com/tracel-ai/burn-central-cli)

## Contribution

Contributions to this repository are welcome. You can also submit issues for features you would like to see in the near future.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

