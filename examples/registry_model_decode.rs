use std::io::Read;

use burn_central_client::BurnCentralCredentials;
use burn_central_core::bundle::{BundleDecode, BundleSource};
use burn_central_registry::RegistryBuilder;

#[derive(Debug, Default)]
struct RawConfig {
    json: String,
}

impl BundleDecode for RawConfig {
    type Settings = ();
    type Error = std::io::Error;

    fn decode<I: BundleSource>(
        source: &I,
        _settings: &Self::Settings,
    ) -> Result<Self, Self::Error> {
        let mut reader = source
            .open("config.json")
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut json = String::new();
        reader.read_to_string(&mut json)?;
        Ok(Self { json })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("BURN_CENTRAL_API_KEY")?;
    let creds = BurnCentralCredentials::new(api_key);

    let registry = RegistryBuilder::new(creds).build()?;

    let model = registry.model("namespace", "project", "model-name")?;
    let cached = model.ensure(1)?;

    println!("Cached model at: {}", cached.path().display());
    println!("Files: {:?}", cached.reader().list()?);

    let config = model.load::<RawConfig>(1, &())?;
    println!("config.json: {}", config.json);

    Ok(())
}
