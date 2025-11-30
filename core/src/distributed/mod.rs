//! # Distributed Training & Inference
//!
//! Welcome to the Distributed module! This is where we break the limits of a single device.
//!
//! ## 🎓 The "Why" of Distributed Computing
//!
//! Imagine you have a book that is too heavy for one person to carry. What do you do?
//! 1.  **Tear the pages out** and give a few pages to each friend (**Data Parallelism**).
//! 2.  **Cut the book in half** down the spine and carry the left half while your friend carries the right (**Tensor Parallelism**).
//! 3.  **Form a bucket brigade**, where you read chapter 1, pass it to a friend for chapter 2, and so on (**Pipeline Parallelism**).
//!
//! In Deep Learning, our "books" are massive matrices (weights) and our "reading" is matrix multiplication.
//! When a model like Gemma or Llama 3 is too big for one GPU's memory, we *must* split it.
//!
//! ## 🧩 Parallelism Strategies
//!
//! ### 1. Data Parallelism (DP)
//! *   **Concept**: Replicate the *entire model* on every device. Split the *dataset*.
//! *   **Analogy**: 8 students solving the same math exam, but each works on different questions. They compare answers at the end.
//! *   **Communication**: Gradients are averaged across devices using **All-Reduce**.
//!
//! ### 2. Tensor Parallelism (TP)
//! *   **Concept**: Split the *model weights* (matrices) across devices.
//! *   **Analogy**: Two pianists playing one piano. One plays the left hand (bass), the other plays the right hand (melody).
//! *   **Communication**: Requires frequent synchronization (All-Reduce, All-Gather) *during* the forward/backward pass.
//! *   **Usage**: Essential for models with billions of parameters (e.g., Megatron-LM).
//!
//! ## 📦 Module Contents
//!
//! *   [`CollectiveBackend`](backend::CollectiveBackend): The interface for communication. We support:
//!     *   **XLA/NCCL**: High-performance GPU communication (Production).
//!     *   **CPU/Ring**: A pure Rust implementation of Ring All-Reduce (Education).
//! *   [`TensorParallelLinear`](crate::distributed::linear::TensorParallelLinear): A Linear layer that automatically splits weights across devices.
//!
//! ## 🚀 Quick Start (Educational)
//!
//! To understand how distributed training works without a GPU cluster, check out [`cpu_backend::CpuBackend`].
//! It implements the **Ring All-Reduce** algorithm from scratch!

pub mod backend;

#[cfg(feature = "xla-backend")]
pub mod comm;
pub mod cpu_backend;
#[cfg(feature = "xla-backend")]
pub mod linear;
