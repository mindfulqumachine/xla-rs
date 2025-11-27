# xla-rs

**xla-rs** is a pedagogical project to build a full LLM training and inference framework in pure Rust, from scratch.

> [!IMPORTANT]
> **We are writing a book!**
> The comprehensive documentation for this project is being written as a **Rustbook**.
> You can find the source in the [`book/`](book/) directory.
> To read it locally:
> ```bash
> mdbook serve book
> ```

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
