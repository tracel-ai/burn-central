use std::io::Read;

#[derive(Debug, thiserror::Error)]
pub enum TransferError {
    #[error("Transport error: {0}")]
    Transport(String),
}

/// Generic client interface used for uploading and downloading files, abstracting over the underlying HTTP client or other transport mechanism.
pub trait FileTransferClient: Clone + Send + Sync + 'static {
    /// Upload data from a reader to the given URL with known size.
    fn put_reader<R: Read + Send + 'static>(
        &self,
        url: &str,
        reader: R,
        size_bytes: u64,
    ) -> Result<(), TransferError>;

    /// Download data from the given URL as a reader.
    fn get_reader(&self, url: &str) -> Result<Box<dyn Read + Send>, TransferError>;
}

/// Reqwest-based transfer client.
#[derive(Clone)]
pub struct ReqwestTransferClient {
    http: reqwest::blocking::Client,
}

impl ReqwestTransferClient {
    pub fn new() -> Self {
        Self {
            http: reqwest::blocking::Client::new(),
        }
    }

    pub fn with_client(http: reqwest::blocking::Client) -> Self {
        Self { http }
    }
}

impl Default for ReqwestTransferClient {
    fn default() -> Self {
        Self::new()
    }
}

impl FileTransferClient for ReqwestTransferClient {
    fn put_reader<R: Read + Send + 'static>(
        &self,
        url: &str,
        reader: R,
        size_bytes: u64,
    ) -> Result<(), TransferError> {
        let body = reqwest::blocking::Body::sized(reader, size_bytes);
        let response = self
            .http
            .put(url)
            .body(body)
            .send()
            .map_err(|e| TransferError::Transport(e.to_string()))?;

        if !response.status().is_success() {
            return Err(TransferError::Transport(
                response.error_for_status().err().unwrap().to_string(),
            ));
        }

        Ok(())
    }

    fn get_reader(&self, url: &str) -> Result<Box<dyn Read + Send>, TransferError> {
        let response = self
            .http
            .get(url)
            .send()
            .map_err(|e| TransferError::Transport(e.to_string()))?;

        if !response.status().is_success() {
            return Err(TransferError::Transport(
                response.error_for_status().err().unwrap().to_string(),
            ));
        }

        Ok(Box::new(response))
    }
}
