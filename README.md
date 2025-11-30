# xla-rs

[![Rust](https://github.com/mindfulqumachine/xla-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/mindfulqumachine/xla-rs/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/mindfulqumachine/xla-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/mindfulqumachine/xla-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Security Audit](https://github.com/mindfulqumachine/xla-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/mindfulqumachine/xla-rs/actions/workflows/audit.yml)

**xla-rs** is a pedagogical project to build a full LLM training and inference framework in pure Rust, from scratch.

> [!IMPORTANT]
> **We are writing a book!**
> The comprehensive documentation for this project is being written as a **Rustbook**.
> You can find the source in the [`book/`](book/) directory.
>
> ### Quick Start
>
> We provide a `Makefile` for common tasks:
>
> ```bash
> # Run the book server + local playground (http://localhost:3000)
> make serve-book
> ```
>
> This command will:
> 1.  Start the local playground server (`local_playground.py`) in the background.
> 2.  Start the `mdBook` server.
>
> ### 2. Access the Book
>
> Open your browser and navigate to `http://localhost:3000`. You should see the book.
>
> ### 3. Run Code Examples
>
> Navigate to any chapter with Rust code examples (e.g., Chapter 2). You should see a "Run" button (a play icon) on the code blocks. Click it to compile and run the code using the local backend.
>
> ### Troubleshooting
>
> If the "Run" button doesn't appear or doesn't work:
>
> 1.  Run `make serve-book`.

## Features

- **Pure Rust**: No C++ dependencies.
- **Tensors**: N-dimensional arrays with broadcasting.
- **Autograd**: Define-by-Run automatic differentiation.
- **Neural Networks**: Linear, RMSNorm, RoPE, Attention, MoE.
- **Gemma**: Full implementation of the Gemma architecture.
- **Zero-Overhead Const Operations**: Perform complex tensor operations like `transpose` and `matmul` entirely at compile time using `ConstDevice`.

> [!NOTE]
> **Implementation Fidelity**
> This project is an **educational implementation** designed to teach the fundamentals of LLMs. It is **not** a faithful reproduction of the official DeepMind Gemma 2 or PaliGemma architectures. Specifically, it currently lacks:
> *   **Interleaved Local-Global Attention**: We use standard global attention for all layers.
> *   **SigLIP Vision Encoders**: We support text-only models, not the multimodal PaliGemma.
> *   **Advanced RL Strategies**: We implement DPO, but not the more complex WARP or BOND strategies used in the official release.

## Roadmap

We are building towards a production-grade system:
- **Compiler**: Graph optimizations and fusion.
- **Training**: Optimizers, Loss functions, Data loading.
- **Serving**: KV Cache, Continuous Batching, Distributed Inference.
- **Advanced**: Diffusion, Mamba, RLHF.

## Linking Code to Chapters

The codebase evolves with the book. To see the code as it corresponds to a specific chapter, use the git tags:

```bash
git checkout chapter-01
# ...
git checkout chapter-05
```

## Appendix: How the Playground Works

The interactive examples in the book are powered by a custom local playground setup. This allows you to run code snippets against your local build of `xla-rs`, rather than waiting for a remote server or relying on the standard Rust Playground which doesn't have access to our local crates.

### Architecture

1.  **`book/theme/custom-playground.js`**: This script intercepts the "Run" button clicks in the mdBook. Instead of sending the code to the official Rust Playground, it sends a POST request to `http://localhost:3001/evaluate.json`.
2.  **`local_playground.py`**: This is a simple Python HTTP server running on port 3001. When it receives code:
    *   It writes the code to a temporary file.
    *   It compiles the code using `rustc`, linking against the `xla_rs` library in your local `target/debug/deps` directory.
    *   It runs the compiled executable and returns the output (stdout/stderr) back to the browser.

This setup ensures that the examples you run in the book are always in sync with your local code changes.
