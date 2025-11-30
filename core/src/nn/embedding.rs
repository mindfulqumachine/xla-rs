use crate::nn::Module;
use crate::tensor::{Cpu, Result, Tensor, TensorElem};

#[derive(Debug, Clone)]
pub struct Embedding<T: TensorElem> {
    pub weight: Tensor<T, 2, Cpu>,
}

impl<T: TensorElem> Embedding<T> {
    pub fn new(weight: Tensor<T, 2, Cpu>) -> Self {
        Self { weight }
    }

    pub fn forward(&self, input: &Tensor<usize, 2, Cpu>) -> Result<Tensor<T, 3, Cpu>> {
        let [batch_size, seq_len] = *input.shape();
        let [vocab_size, hidden_dim] = *self.weight.shape();

        let input_data = input.data();
        let weight_data = self.weight.data();

        // Output: [batch, seq, hidden]
        let mut out = Tensor::zeros([batch_size, seq_len, hidden_dim]);
        let out_data = out.data_mut();

        // TODO: Parallelize this loop using rayon if needed
        for (i, &token_id) in input_data.iter().enumerate() {
            if token_id >= vocab_size {
                return Err(crate::tensor::TensorError::IndexOutOfBounds {
                    index: vec![token_id],
                    shape: vec![vocab_size, hidden_dim],
                });
            }

            let src_start = token_id * hidden_dim;
            let dest_start = i * hidden_dim;

            out_data[dest_start..dest_start + hidden_dim]
                .copy_from_slice(&weight_data[src_start..src_start + hidden_dim]);
        }

        Ok(out)
    }
}

impl<T: TensorElem> Module<T> for Embedding<T> {}
