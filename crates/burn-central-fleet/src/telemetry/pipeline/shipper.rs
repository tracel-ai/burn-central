use std::sync::Arc;
use std::sync::mpsc::{RecvTimeoutError, Sender, channel};
use std::time::Duration;

use super::super::envelope::TelemetryEnvelope;

use super::Outbox;

pub trait ShipperTransport: Send + Sync {
    fn ship(&self, data: Vec<TelemetryEnvelope>) -> Result<(), String>;
}

#[derive(Default)]
pub struct NoopShipperTransport;

impl ShipperTransport for NoopShipperTransport {
    fn ship(&self, _data: Vec<TelemetryEnvelope>) -> Result<(), String> {
        Ok(())
    }
}

pub struct ShipperHandle {
    join_handle: Option<std::thread::JoinHandle<()>>,
    shutdown_tx: Option<Sender<()>>,
}

impl ShipperHandle {
    pub fn shutdown(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        if let Some(join_handle) = self.join_handle.take() {
            if join_handle.join().is_err() {
                tracing::warn!("telemetry shipper thread panicked during shutdown");
            }
        }
    }
}

impl Drop for ShipperHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub fn start(
    outbox: Arc<dyn Outbox>,
    transport: Arc<dyn ShipperTransport>,
    interval: Duration,
) -> ShipperHandle {
    let (shutdown_tx, shutdown_rx) = channel::<()>();
    let join_handle = std::thread::Builder::new()
        .name("telemetry-shipper".to_string())
        .spawn(move || {
            loop {
                match outbox.claim(10) {
                    Ok(items) => {
                        let (ids, envelopes): (Vec<_>, Vec<_>) = items.into_iter().unzip();
                        match transport.ship(envelopes) {
                            Ok(_) => {
                                for id in ids {
                                    if let Err(e) = outbox.complete(id) {
                                        tracing::error!(
                                            "failed to complete telemetry outbox item {id}: {e}"
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!("failed to ship telemetry batch: {e}");
                                for id in ids {
                                    if let Err(err) = outbox.fail(id, &e) {
                                        tracing::error!(
                                            "failed to mark telemetry outbox item {id} as failed: {err}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("failed to claim telemetry outbox items: {e}");
                    }
                }

                match shutdown_rx.recv_timeout(interval) {
                    Ok(_) | Err(RecvTimeoutError::Disconnected) => break,
                    Err(RecvTimeoutError::Timeout) => {}
                }
            }
        })
        .expect("failed to spawn shipper thread");

    ShipperHandle {
        join_handle: Some(join_handle),
        shutdown_tx: Some(shutdown_tx),
    }
}
