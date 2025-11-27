use crate::tensor::{Cpu, Result, Tensor, TensorElem};
use num_traits::Float;
use rayon::prelude::*;

/// RMSNorm (Root Mean Square Layer Normalization).
///
/// Normalizes the input tensor using the root mean square of the elements.
/// Used in modern LLM architectures like Gemma and Llama.
#[derive(Debug)]
pub struct RMSNorm<T: TensorElem> {
    pub weight: Tensor<T, 1, Cpu>,
    pub eps: T,
}

impl<T: TensorElem + Float> RMSNorm<T> {
    pub fn new(weight: Tensor<T, 1, Cpu>, eps: T) -> Self {
        Self { weight, eps }
    }

    pub fn forward<const RANK: usize>(
        &self,
        x: &Tensor<T, RANK, Cpu>,
    ) -> Result<Tensor<T, RANK, Cpu>> {
        let shape = x.shape();
        let last_dim = shape[RANK - 1];
        if last_dim != self.weight.shape()[0] {
            return Err(crate::tensor::TensorError::ShapeMismatch {
                expected: vec![last_dim],
                got: vec![self.weight.shape()[0]],
            });
        }

        let mut out = Tensor::zeros(*shape);

        out.data_mut()
            .par_chunks_mut(last_dim)
            .zip(x.data().par_chunks(last_dim))
            .for_each(|(out_row, in_row)| {
                let mut sum_sq = T::zero();
                for &val in in_row {
                    sum_sq += val * val;
                }
                let mean_sq = sum_sq / T::from_usize(last_dim).unwrap();
                let rsqrt = T::one() / (mean_sq + self.eps).sqrt();

                for i in 0..last_dim {
                    out_row[i] = in_row[i] * rsqrt * self.weight.data()[i];
                }
            });

        Ok(out)
    }
}
