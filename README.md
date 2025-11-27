# xla-rs

[![Rust](https://github.com/mindfulqumachine/xla-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/mindfulqumachine/xla-rs/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/mindfulqumachine/xla-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/mindfulqumachine/xla-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
> make serve
>
> # Run book tests
> make test-book
> ```
>
> To fully enjoy the interactive examples:
> 1.  Run `make serve`.
> 2.  Navigate to `http://localhost:3000`.

## Features

- **Pure Rust**: No C++ dependencies.
- **Tensors**: N-dimensional arrays with broadcasting.
- **Autograd**: Define-by-Run automatic differentiation.
- **Neural Networks**: Linear, RMSNorm, RoPE, Attention, MoE.
- **Gemma**: Full implementation of the Gemma architecture.

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
