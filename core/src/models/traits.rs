use crate::tensor::{Cpu, Result, Tensor, TensorElem};
use std::path::Path;

/// A trait for Causal Language Models.
///
/// # What is a Causal LM?
///
/// A Causal Language Model predicts the next token in a sequence based only on previous tokens.
/// It uses a "causal mask" (lower triangular matrix) in attention to prevent "cheating" by looking ahead.
///
/// Examples: GPT-2, Llama, Gemma.
///
/// # Interface
///
/// This trait defines the standard interface for:
/// - **Forward Pass**: Computing logits from input IDs.
/// - **Generation**: Auto-regressively generating text.
/// - **Loading**: Loading weights from disk.
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

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyLM;

    impl CausalLM<f32> for DummyLM {
        fn forward(&self, _input_ids: &Tensor<usize, 2, Cpu>) -> Result<Tensor<f32, 3, Cpu>> {
            Ok(Tensor::zeros([1, 1, 1]))
        }

        fn load_weights<P: AsRef<Path>>(_path: P) -> Result<Self> {
            Ok(DummyLM)
        }
    }

    #[test]
    fn test_default_generate_implementation() {
        let model = DummyLM;
        let prompt = Tensor::new(vec![0usize], [1, 1]).unwrap();
        let result = model.generate(&prompt, 10);
        assert!(result.is_err());
        match result {
            Err(crate::tensor::TensorError::Unsupported(msg)) => {
                assert_eq!(msg, "Generation not implemented");
            }
            _ => panic!("Expected Unsupported error"),
        }
    }
}
