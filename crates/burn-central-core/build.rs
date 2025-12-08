fn main() {
    println!("cargo::rerun-if-env-changed=DEP_BURN_VERSION");
    println!("cargo::rerun-if-env-changed=DEP_BURN_VERSION_MAJOR_MINOR");

    // These will be set by Cargo because burn uses 'links'
    if let Ok(burn_version) = std::env::var("DEP_BURN_VERSION") {
        println!("cargo::warning=Detected Burn version: {}", burn_version);

        // Parse and set appropriate feature
        let version_parts: Vec<&str> = burn_version.split('.').collect();
        match (version_parts[0], version_parts[1]) {
            ("0", "19") => {
                println!("cargo::rustc-cfg=burn_0_19");
                println!("cargo::warning=Enabling burn_0_19 feature");
            }
            ("0", "20") => {
                println!("cargo::rustc-cfg=burn_0_20");
                println!("cargo::warning=Enabling burn_0_20 feature");
            }
            _ => panic!("Unsupported Burn version: {}", burn_version),
        }
    } else {
        println!(
            "cargo::warning=No version of burn detected defaulting to most recent supported version"
        );
        println!("cargo::rustc-cfg=burn_0_20");
    }
}

