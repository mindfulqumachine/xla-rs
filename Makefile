.PHONY: build test lint book serve playground test-book clean

# Default target
all: build lint test test-book

# Rust commands
build:
	cargo build

test:
	cargo test

lint:
	cargo clippy -- -D warnings

clean:
	cargo clean
	rm -rf book/book

# Book commands
book: serve

serve:
	@echo "Starting local playground and mdbook..."
	@echo "The playground server runs in the background. Press Ctrl+C to stop both."
	@trap 'kill %1' SIGINT EXIT; \
	python3 local_playground.py & \
	mdbook serve book

playground:
	@echo "Starting local playground server on port 3001..."
	python3 local_playground.py

test-book: build
	cargo clean -p xla_rs
	cargo build
	@echo "Running mdbook tests..."
	mdbook test -L target/debug/deps book
