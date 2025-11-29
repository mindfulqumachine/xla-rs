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
    /// Creates a new RMSNorm layer.
    ///
    /// # Arguments
    ///
    /// * `weight` - The scale weights of shape `[features]`.
    /// * `eps` - A small constant for numerical stability.
    pub fn new(weight: Tensor<T, 1, Cpu>, eps: T) -> Self {
        Self { weight, eps }
    }

    /// Performs the forward pass of RMSNorm.
    ///
    /// Normalizes the input over the last dimension.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_rmsnorm_forward() {
        // Input: [1.0, 2.0, 3.0]
        // RMS = sqrt((1+4+9)/3) = sqrt(14/3) = sqrt(4.666) approx 2.1602
        // Norm: [1/2.16, 2/2.16, 3/2.16] -> [0.4629, 0.9258, 1.3887]
        // Weight: [1, 1, 1] -> Output same as norm

        let input_data = vec![1.0, 2.0, 3.0];
        let input = Tensor::<f32, 1, Cpu>::new(input_data, [3]).unwrap();

        let weight = Tensor::<f32, 1, Cpu>::ones([3]);
        let norm = RMSNorm::new(weight, 1e-5); // Small eps

        let output = norm.forward(&input).unwrap();
        let out_data = output.data();

        // Expected values calculation
        let rms = (14.0f32 / 3.0).sqrt();
        let expected = vec![1.0 / rms, 2.0 / rms, 3.0 / rms];

        for (got, exp) in out_data.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-4);
        }
    }

    #[test]
    fn test_rmsnorm_shape_mismatch() {
        let weight = Tensor::<f32, 1, Cpu>::ones([4]); // Expect last dim 4
        let norm = RMSNorm::new(weight, 1e-5);

        let input = Tensor::<f32, 2, Cpu>::zeros([2, 3]); // Last dim 3
        let res = norm.forward(&input);
        assert!(res.is_err());
    }
}
