//! Pre-built Model Architectures.
//!
//! This module contains implementations of popular Large Language Models (LLMs).
//! These implementations are designed to be readable and educational, mirroring the structure
//! of the original papers/codebases where possible.
//!
//! # Available Models
//!
//! - **Gemma**: Google's open weights LLM (2B and 7B variants). See [`gemma`].
//! - **GPT-2**: OpenAI's classic transformer model. See [`gpt2`].
//!
//! # Example: Loading and Running Gemma
//!
//! ```rust, no_run
//! use xla_rs::models::gemma::{GemmaForCausalLM, GemmaConfig};
//! use xla_rs::models::traits::CausalLM;
//! use xla_rs::tensor::Tensor;
//!
//! // 1. Configure the model (e.g., Gemma 2B)
//! let config = GemmaConfig::gemma_2b();
//!
//! // 2. Instantiate the model (weights would be loaded here in a real scenario)
//! // Note: This example just creates a model with random/empty weights for demonstration.
//! let model = GemmaForCausalLM::new(config).unwrap();
//!
//! // 3. Create a dummy input (batch_size=1, seq_len=10)
//! // In reality, this would be token IDs from a tokenizer.
//! let input_ids = Tensor::<i64, 2>::zeros([1, 10]);
//!
//! // 4. Forward pass
//! // Returns logits: [Batch, Seq, Vocab]
//! let logits = model.forward(&input_ids).unwrap();
//!
//! assert_eq!(logits.shape(), &[1, 10, 256000]); // Gemma vocab size
//! ```

pub mod gemma;
pub mod gpt2;
pub mod traits;
