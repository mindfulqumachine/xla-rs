//! # xla-rs
//!
//! `xla-rs` is a pure Rust implementation of tensor operations and neural network building blocks,
//! designed for educational purposes and understanding the internals of LLM inference.
//!
//! Despite the name, it currently runs on **CPU only** and does not yet integrate with the XLA compiler.
//!
//! ## Modules
//!
//! - [`tensor`]: Core N-dimensional tensor implementation.
//! - [`nn`]: Neural network layers (Linear, RMSNorm, MoE, etc.).
//! - [`models`]: Model architectures (e.g., Gemma).
//!
//! ## Example
//!
//! ```rust
//! use xla_rs::tensor::Tensor;
//!
//! let data = vec![1.0, 2.0, 3.0, 4.0];
//! let tensor = Tensor::<f32, 2>::new(data, [2, 2]).unwrap();
//! println!("{:?}", tensor);
//! ```

pub mod autograd;
pub mod models;
pub mod nn;
pub mod tensor;

pub use tensor::Tensor;
