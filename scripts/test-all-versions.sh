#!/bin/bash
set -e

echo "========================================="
echo "Testing burn-central-core with Burn 0.20"
echo "========================================="
cargo test -p burn-central-core --features burn_0_20 --no-default-features

echo ""
echo "========================================="
echo "Testing burn-central-core with Burn 0.19"
echo "========================================="
cargo test -p burn-central-core --features burn_0_19 --no-default-features

echo ""
echo "========================================="
echo "✓ All versions tested successfully!"
echo "========================================="
