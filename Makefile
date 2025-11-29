.PHONY: build test lint book serve playground test-book clean doc doc-test serve-doc

# Default target
all: fmt build lint test test-book doc doc-test

# Rust commands
build:
	cargo build

test:
	cargo test

doc:
	cargo doc --no-deps

doc-test:
	cargo test --doc

serve-doc: doc
	@echo "Serving API documentation on http://localhost:8000"
	cd target/doc && python3 -m http.server 8000

lint:
	cargo clippy -- -D warnings

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clean:
	cargo clean
	rm -rf book/book

# Book commands
book: serve-book

serve-book:
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

coverage:
	@if ! cargo --list | grep -q "tarpaulin"; then \
		echo "cargo-tarpaulin not found. Installing..."; \
		cargo install cargo-tarpaulin; \
	fi
	cargo tarpaulin --workspace --fail-under 98 --out Xml --out Html

spellcheck:
	@if ! command -v typos >/dev/null 2>&1; then \
		echo "typos not found. Installing..."; \
		cargo install typos-cli; \
	fi
	typos

ci: build lint test test-book doc-test coverage spellcheck
