use crate::experiment::CancelToken;
use crate::experiment::log_store::TempLogStore;
use burn_central_client::{
    ClientError, WebSocketClient,
    websocket::{ExperimentMessage, ServerMessage},
};
use crossbeam::channel::{Receiver, RecvTimeoutError, TryRecvError};
use std::{thread::JoinHandle, time::Duration};

#[derive(Debug, thiserror::Error)]
pub enum ThreadError {
    #[error("WebSocket error: {0}")]
    WebSocket(String),
    #[error("Log storage failed: {0}")]
    LogFlushError(ClientError),
    #[error("Unexpected panic in thread")]
    Panic,
}

const WEBSOCKET_CLOSE_ERROR: &str = "Failed to close WebSocket";

#[derive(Debug)]
pub struct ThreadResult {}

struct ExperimentThread {
    ws_client: WebSocketClient,
    message_receiver: Receiver<ExperimentMessage>,
    log_store: TempLogStore,
    cancel_token: CancelToken,
}

impl ExperimentThread {
    fn new(
        ws_client: WebSocketClient,
        message_receiver: Receiver<ExperimentMessage>,
        log_store: TempLogStore,
        cancel_token: CancelToken,
    ) -> Self {
        Self {
            ws_client,
            message_receiver,
            log_store,
            cancel_token,
        }
    }

    fn run(mut self) -> Result<ThreadResult, ThreadError> {
        let res = self.thread_loop();
        self.cleanup()?;
        res.map(|_| ThreadResult {})
    }

    fn cleanup(&mut self) -> Result<(), ThreadError> {
        self.ws_client
            .close()
            .map_err(|_| ThreadError::WebSocket(WEBSOCKET_CLOSE_ERROR.to_string()))?;
        self.log_store.flush().map_err(ThreadError::LogFlushError)?;
        self.ws_client
            .wait_until_closed()
            .map_err(|e| ThreadError::WebSocket(e.to_string()))?;
        Ok(())
    }

    fn handle_websocket_send<T: serde::Serialize + std::fmt::Debug>(
        &mut self,
        message: T,
    ) -> Result<(), ThreadError> {
        self.ws_client
            .send(message)
            .map_err(|e| ThreadError::WebSocket(e.to_string()))
    }

    fn handle_log_message(&mut self, log: String) -> Result<(), ThreadError> {
        self.log_store
            .push(log.clone())
            .map_err(ThreadError::LogFlushError)?;
        self.handle_websocket_send(ExperimentMessage::Log(log))
    }

    fn process_message(&mut self, message: ExperimentMessage) -> Result<(), ThreadError> {
        match message {
            ExperimentMessage::MetricsLog { .. } => {
                self.handle_websocket_send(message)?;
            }
            ExperimentMessage::MetricDefinitionLog { .. } => {
                self.handle_websocket_send(message)?;
            }
            ExperimentMessage::Log(log) => {
                self.handle_log_message(log)?;
            }
            ExperimentMessage::EpochSummaryLog { .. } => {
                self.handle_websocket_send(message)?;
            }
            value => {
                self.handle_websocket_send(value)?;
            }
        }
        Ok(())
    }

    fn thread_loop(&mut self) -> Result<(), ThreadError> {
        let poll = Duration::from_millis(50);

        loop {
            match self.ws_client.receive::<ServerMessage>() {
                Ok(Some(ServerMessage::CancelExperiment { reason })) => {
                    tracing::info!(%reason, "Received server cancel request - triggering cancellation token");
                    self.cancel_token.cancel();
                }
                Ok(None) => {}
                Err(e) => tracing::error!(error = ?e, "WebSocket receive error"),
            }

            match self.message_receiver.recv_timeout(poll) {
                Ok(message) => self.process_message(message)?,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct ExperimentSocket {
    handle: JoinHandle<Result<ThreadResult, ThreadError>>,
}

impl ExperimentSocket {
    pub fn new(
        ws_client: WebSocketClient,
        log_store: TempLogStore,
        message_receiver: Receiver<ExperimentMessage>,
        cancel_token: CancelToken,
    ) -> Self {
        let thread = ExperimentThread::new(ws_client, message_receiver, log_store, cancel_token);
        let handle = std::thread::spawn(move || thread.run());
        Self { handle }
    }

    pub fn join(self) -> Result<ThreadResult, ThreadError> {
        self.handle.join().unwrap_or(Err(ThreadError::Panic))
    }
}
