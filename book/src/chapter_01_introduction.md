# Chapter 1: Introduction & Philosophy

> "What I cannot create, I do not understand." — Richard Feynman

Welcome to **xla-rs**. This book is a journey through the internals of Large Language Models (LLMs). We aren't just going to use them; we are going to build one, from scratch, in Rust.

## Philosophy

Our approach is guided by three principles:

1.  **Pure Rust**: We avoid binding to C++ libraries like Torch or TensorFlow. We want to understand the entire stack, down to the memory layout of a tensor.
2.  **Pedantic Implementation**: We prioritize clarity over raw performance (initially). We want the code to map 1:1 with the mathematical equations.
3.  **From Scratch**: We start with `Vec<f32>` and end with a distributed inference server.

## Prerequisites

You will need:
- **Rust**: Latest stable version (`rustup update stable`).
- **A Curiosity for Math**: We will cover the necessary linear algebra and calculus as we go.

## Setting Up

Clone the repository and run the setup test to ensure everything is working.

```bash
git clone https://github.com/mindfulqumachine/xla-rs.git
cd xla-rs
cargo test --test chapter_01_setup
```

If the test passes, you are ready to begin!
