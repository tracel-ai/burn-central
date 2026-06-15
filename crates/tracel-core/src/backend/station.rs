use tracel_client::StationClient;
use url::Url;

#[derive(Debug, Clone)]
pub struct StationBackend {
    pub client: StationClient,
}

impl StationBackend {
    pub fn create_context(url: Url) -> StationBackend {
        StationBackend {
            client: StationClient::from_url(url),
        }
    }
}
