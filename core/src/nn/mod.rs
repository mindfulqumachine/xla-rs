//! Neural Network Building Blocks.
//!
//! # What is a Module?
//!
//! In Deep Learning, a "Module" (or Layer) is a stateful container that encapsulates:
//! 1. **Parameters**: Learnable weights (e.g., `Linear` weights, `LayerNorm` bias).
//! 2. **Forward Pass**: The logic to transform input tensors into output tensors.
//!
//! This module provides the standard building blocks for creating neural networks.
//!
//! # Available Layers
//!
//! - **Linear**: Fully connected layer ($y = xA^T + b$).
//! - **Normalization**: `RMSNorm` (Gemma/Llama) and `LayerNorm` (GPT-2/BERT).
//! - **Embedding**: Lookup table for converting tokens to vectors.
//! - **Activation**: Non-linear functions like ReLU, GELU.
//! - **Transformer**: Multi-Head Attention and Transformer Blocks.

pub mod activation;
pub mod embedding;
pub mod linear;
pub mod lora;
pub mod module;
pub mod norm;
pub mod transformer;

pub mod conv;
pub mod pool;

pub use activation::Activation;
pub use conv::Conv2d;
pub use embedding::Embedding;
pub use linear::{AllowedLinearRank, Linear};
pub use module::Module;
pub use norm::{LayerNorm, RMSNorm};
pub use pool::MaxPool2d;
