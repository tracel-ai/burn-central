#!/bin/bash

# Script to continuously regenerate cargo docs on file changes
# Requires cargo-watch: cargo install cargo-watch

set -e

echo "Starting cargo doc watcher..."
echo "Documentation will be regenerated on file changes."
echo ""
echo "To install cargo-watch if you don't have it:"
echo "  cargo install cargo-watch"
echo ""

# Check if cargo-watch is installed
if ! command -v cargo-watch &>/dev/null; then
	echo "Error: cargo-watch is not installed."
	echo "Install it with: cargo install cargo-watch"
	exit 1
fi

# Run cargo doc once and open in browser
echo "Building docs initially..."
cargo doc --no-deps --open

# Watch for changes and rebuild docs
# -x: Execute cargo command
# -w: Watch specific paths (src and Cargo.toml)
# -c: Clear screen before each run
cargo-watch \
	-w crates \
	-w Cargo.toml \
	-c \
	-x 'doc --no-deps'
