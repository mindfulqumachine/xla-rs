//! Transformer Building Blocks.
//!
//! This module contains the core components of the Transformer architecture, which powers
//! modern Large Language Models (LLMs).
//!
//! # Components
//!
//! - **Attention**: The mechanism that allows the model to "attend" to different parts of the sequence.
//!   - [`MultiHeadAttention`](attention::MultiHeadAttention): Standard implementation.
//!   - [`OptimizedMultiHeadAttention`](attention_optimized::OptimizedMultiHeadAttention): Fused/Optimized version.
//! - **RoPE**: Rotary Positional Embeddings (encodes position information).
//! - **MoE**: Mixture of Experts (sparse activation).

pub mod attention;
pub mod attention_optimized;
pub mod moe;
pub mod rope;
