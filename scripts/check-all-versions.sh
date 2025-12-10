#!/bin/bash
set -e

echo "========================================="
echo "Checking burn-central-core with Burn 0.20"
echo "========================================="
cargo check -p burn-central-core --features burn_0_20 --no-default-features

echo ""
echo "========================================="
echo "Checking burn-central-core with Burn 0.19"
echo "========================================="
cargo check -p burn-central-core --features burn_0_19 --no-default-features

echo ""
echo "========================================="
echo "✓ All versions checked successfully!"
echo "========================================="
