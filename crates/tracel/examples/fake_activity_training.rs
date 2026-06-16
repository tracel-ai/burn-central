use std::collections::BTreeMap;
use std::error::Error;
use std::thread;
use std::time::Duration;

use tracel::experiment::{ExperimentRun, MetricSpec, MetricValue};
use tracel::{Connection, Context};

const EPOCHS: usize = 3;
const BATCHES_PER_EPOCH: usize = 8;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let context = Context::new(Connection::Cloud)?;

    context
        .experiment()
        .create(
            "fake-activity-training",
            fake_activity_training_with_evaluation,
        )
        .attribute("kind", "example")?
        .attribute("dataset", "synthetic-sleep")?
        .attribute("epochs", EPOCHS)?
        .attribute("batches_per_epoch", BATCHES_PER_EPOCH)?
        .run(())
}

fn fake_activity_training_with_evaluation(
    experiment: &ExperimentRun,
    _input: (),
) -> Result<(), Box<dyn Error + Send + Sync>> {
    fake_training(experiment, ())?;
    fake_evaluation(experiment, ())?;
    Ok(())
}

fn fake_training(
    experiment: &ExperimentRun,
    _input: (),
) -> Result<(), Box<dyn Error + Send + Sync>> {
    experiment.log_info("starting fake activity training")?;
    experiment.log_config("training", &training_config())?;
    log_metric_definitions(experiment)?;

    let mut training = experiment
        .activity("training")
        .progress()
        .total(EPOCHS as u64)
        .unit("epoch")
        .cancellable()
        .attr("model", "tiny-mlp")?
        .start();

    let total_batches = (EPOCHS * BATCHES_PER_EPOCH) as f64;

    for epoch in 1..=EPOCHS {
        let mut epoch_activity = training
            .activity(format!("epoch {epoch}"))
            .progress()
            .total(BATCHES_PER_EPOCH as u64)
            .unit("batch")
            .cancellable()
            .attr("epoch", epoch)?
            .start();

        let mut loss_sum = 0.0;
        let mut accuracy_sum = 0.0;

        for batch in 1..=BATCHES_PER_EPOCH {
            let batch_activity = epoch_activity
                .activity(format!("batch {batch}"))
                .cancellable()
                .attr("epoch", epoch)?
                .attr("batch", batch)?
                .start();

            if batch_activity.is_cancel_requested() {
                batch_activity.abandon_with_message("cancel requested");
                epoch_activity.abandon_with_message("cancel requested");
                training.abandon_with_message("cancel requested");
                return Ok(());
            }

            thread::sleep(Duration::from_millis(1010));

            let step = ((epoch - 1) * BATCHES_PER_EPOCH + batch) as f64;
            let loss = 1.2 / (1.0 + step * 0.32);
            let accuracy = (0.48 + step / total_batches * 0.45).min(0.97);

            experiment.log_metric(
                epoch,
                "train",
                batch,
                vec![
                    MetricValue {
                        name: "loss".to_string(),
                        value: loss,
                    },
                    MetricValue {
                        name: "accuracy".to_string(),
                        value: accuracy,
                    },
                ],
            )?;

            loss_sum += loss;
            accuracy_sum += accuracy;
            epoch_activity.inc(1);
            batch_activity.finish_with_message(format!("loss={loss:.3}, accuracy={accuracy:.3}"));
        }

        let mean_loss = loss_sum / BATCHES_PER_EPOCH as f64;
        let mean_accuracy = accuracy_sum / BATCHES_PER_EPOCH as f64;

        experiment.log_epoch_summary(
            epoch,
            "train",
            vec![
                MetricValue {
                    name: "loss".to_string(),
                    value: mean_loss,
                },
                MetricValue {
                    name: "accuracy".to_string(),
                    value: mean_accuracy,
                },
            ],
        )?;

        epoch_activity.finish_with_message(format!(
            "mean_loss={mean_loss:.3}, mean_accuracy={mean_accuracy:.3}"
        ));
        training.inc(1);
    }

    training.finish_with_message("fake training complete");
    experiment.log_info("finished fake activity training")?;

    Ok(())
}

fn fake_evaluation(
    experiment: &ExperimentRun,
    _input: (),
) -> Result<(), Box<dyn Error + Send + Sync>> {
    experiment.log_info("starting fake evaluation")?;

    let mut evaluation = experiment
        .activity("evaluation")
        .progress()
        .total(5)
        .unit("tests")
        .cancellable()
        .attr("model", "tiny-mlp")?
        .start();

    for test in 1..=5 {
        let test_activity = evaluation
            .activity(format!("test {test}"))
            .cancellable()
            .attr("test", test)?
            .start();

        if test_activity.is_cancel_requested() {
            test_activity.abandon_with_message("cancel requested");
            evaluation.abandon_with_message("cancel requested");
            return Ok(());
        }

        thread::sleep(Duration::from_millis(1500));
        let test_accuracy = 0.5 + test as f64 * 0.09;
        experiment.log_metric(
            test,
            "evaluation",
            0,
            vec![MetricValue {
                name: "accuracy".to_string(),
                value: test_accuracy,
            }],
        )?;
        test_activity.finish_with_message(format!("accuracy={test_accuracy:.3}"));
        evaluation.inc(1);
    }

    evaluation.finish_with_message("evaluation complete");
    experiment.log_info("finished fake evaluation")?;

    Ok(())
}

fn training_config() -> BTreeMap<&'static str, usize> {
    BTreeMap::from([
        ("epochs", EPOCHS),
        ("batches_per_epoch", BATCHES_PER_EPOCH),
        ("hidden_size", 32),
        ("sleep_ms_per_batch", 150),
    ])
}

fn log_metric_definitions(experiment: &ExperimentRun) -> Result<(), Box<dyn Error + Send + Sync>> {
    experiment.log_metric_definition(MetricSpec {
        name: "loss".to_string(),
        description: Some("Fake training loss that decreases over time".to_string()),
        unit: None,
        higher_is_better: false,
    })?;
    experiment.log_metric_definition(MetricSpec {
        name: "accuracy".to_string(),
        description: Some("Fake accuracy that increases over time".to_string()),
        unit: Some("ratio".to_string()),
        higher_is_better: true,
    })?;

    Ok(())
}
