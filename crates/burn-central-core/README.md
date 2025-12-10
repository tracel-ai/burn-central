# burn-central-core

Core library for Burn Central, providing experiment tracking and logging capabilities compatible with Burn Central for burn model.

## Multi-Version Burn Support

burn-central-core 0.3+ supports both Burn 0.19 and 0.20 through feature flags, allowing a single burn-central version to work with multiple Burn versions.

### Usage

**Burn 0.20 (default):**
```toml
[dependencies]
burn-central-core = "0.3"
```

**Burn 0.19:**
```toml
[dependencies]
burn-central-core = { version = "0.3", default-features = false, features = ["burn_0_19"] }
```

### Important Notes

1. **Mutually Exclusive**: You cannot use both versions simultaneously. Choose one based on your project's Burn version.

2. **Burn 0.19 Limitations**: The Burn 0.19 implementation provides stub implementations for some features:
   - `RemoteMetricLogger`: Provides a no-op implementation that satisfies trait bounds but doesn't perform actual metric logging to Burn Central.
   - For full functionality, please use Burn 0.20 or later.

3. **Type Re-exports**: Always import Burn types through `burn_central_core::burn::*` to ensure version compatibility across your codebase.

### Development

```bash
# Check/test Burn 0.20
cargo check -p burn-central --features burn_0_20 --no-default-features
cargo test -p burn-central --features burn_0_20 --no-default-features

# Check/test Burn 0.19
cargo check -p burn-central --features burn_0_19 --no-default-features
cargo test -p burn-central --features burn_0_19 --no-default-features
```
