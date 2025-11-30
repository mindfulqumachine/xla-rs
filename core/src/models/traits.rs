use crate::tensor::{Cpu, Result, Tensor, TensorElem};
use std::path::Path;

/// A trait for Causal Language Models (e.g., GPT, Llama, Gemma).
///
/// This trait defines the standard interface for interacting with language models
/// in `xla-rs`, ensuring consistency across different architectures.
pub trait CausalLM<T: TensorElem> {
    /// Performs a forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - A tensor of shape [batch_size, sequence_length] containing token IDs.
    ///
    /// # Returns
    ///
    /// * `logits` - A tensor of shape [batch_size, sequence_length, vocab_size].
    fn forward(&self, input_ids: &Tensor<usize, 2, Cpu>) -> Result<Tensor<T, 3, Cpu>>;

    /// Generates text from a prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt_ids` - A tensor of shape [batch_size, sequence_length] containing prompt token IDs.
    /// * `max_len` - The maximum length of the generated sequence.
    ///
    /// # Returns
    ///
    /// * `output_ids` - A tensor of shape [batch_size, total_length] containing the generated token IDs.
    fn generate(
        &self,
        prompt_ids: &Tensor<usize, 2, Cpu>,
        max_len: usize,
    ) -> Result<Tensor<usize, 2, Cpu>> {
        // TODO: Implement default greedy generation
        let _ = (prompt_ids, max_len);
        Err(crate::tensor::TensorError::Unsupported(
            "Generation not implemented".to_string(),
        ))
    }

    /// Loads weights from a file (e.g., Safetensors).
    fn load_weights<P: AsRef<Path>>(path: P) -> Result<Self>
    where
        Self: Sized;
}
