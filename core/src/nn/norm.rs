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

/// Layer Normalization.
///
/// Normalizes the input tensor using the mean and standard deviation of the elements.
/// Used in architectures like GPT-2, BERT, etc.
/// Formula: `y = (x - mean) / sqrt(var + eps) * gamma + beta`
#[derive(Debug)]
pub struct LayerNorm<T: TensorElem> {
    pub weight: Tensor<T, 1, Cpu>,
    pub bias: Tensor<T, 1, Cpu>,
    pub eps: T,
}

impl<T: TensorElem + Float> LayerNorm<T> {
    /// Creates a new LayerNorm layer.
    ///
    /// # Arguments
    ///
    /// * `weight` - The scale weights (gamma) of shape `[features]`.
    /// * `bias` - The shift weights (beta) of shape `[features]`.
    /// * `eps` - A small constant for numerical stability.
    pub fn new(weight: Tensor<T, 1, Cpu>, bias: Tensor<T, 1, Cpu>, eps: T) -> Self {
        Self { weight, bias, eps }
    }

    /// Performs the forward pass of LayerNorm.
    ///
    /// Normalizes the input over the last dimension.
    pub fn forward<const RANK: usize>(
        &self,
        x: &Tensor<T, RANK, Cpu>,
    ) -> Result<Tensor<T, RANK, Cpu>> {
        let shape = x.shape();
        let last_dim = shape[RANK - 1];
        if last_dim != self.weight.shape()[0] || last_dim != self.bias.shape()[0] {
            return Err(crate::tensor::TensorError::ShapeMismatch {
                expected: vec![last_dim, last_dim],
                got: vec![self.weight.shape()[0], self.bias.shape()[0]],
            });
        }

        let mut out = Tensor::zeros(*shape);

        out.data_mut()
            .par_chunks_mut(last_dim)
            .zip(x.data().par_chunks(last_dim))
            .for_each(|(out_row, in_row)| {
                let mut sum = T::zero();
                for &val in in_row {
                    sum += val;
                }
                let mean = sum / T::from_usize(last_dim).unwrap();

                let mut sum_sq_diff = T::zero();
                for &val in in_row {
                    let diff = val - mean;
                    sum_sq_diff += diff * diff;
                }
                let var = sum_sq_diff / T::from_usize(last_dim).unwrap();
                let rstd = T::one() / (var + self.eps).sqrt();

                for i in 0..last_dim {
                    out_row[i] =
                        (in_row[i] - mean) * rstd * self.weight.data()[i] + self.bias.data()[i];
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
    fn test_layernorm_forward() {
        // Input: [1.0, 2.0, 3.0]
        // Mean: 2.0
        // Var: ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1 + 0 + 1) / 3 = 2/3
        // Std: sqrt(2/3) approx 0.8165
        // Norm: [(1-2)/0.8165, 0, (3-2)/0.8165] -> [-1.2247, 0.0, 1.2247]
        // Weight: [1, 1, 1], Bias: [0, 0, 0] -> Output same as norm

        let input_data = vec![1.0, 2.0, 3.0];
        let input = Tensor::<f32, 1, Cpu>::new(input_data, [3]).unwrap();

        let weight = Tensor::<f32, 1, Cpu>::ones([3]);
        let bias = Tensor::<f32, 1, Cpu>::zeros([3]);
        let norm = LayerNorm::new(weight, bias, 1e-5);

        let output = norm.forward(&input).unwrap();
        let out_data = output.data();

        // Expected values
        let mean = 2.0;
        let var = 2.0 / 3.0;
        let std = (var + 1e-5).sqrt();
        let expected = vec![(1.0 - mean) / std, (2.0 - mean) / std, (3.0 - mean) / std];

        for (got, exp) in out_data.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-4);
        }
    }

    #[test]
    fn test_layernorm_shape_mismatch() {
        let weight = Tensor::<f32, 1, Cpu>::ones([4]);
        let bias = Tensor::<f32, 1, Cpu>::zeros([4]);
        let norm = LayerNorm::new(weight, bias, 1e-5);

        let input = Tensor::<f32, 2, Cpu>::zeros([2, 3]);
        let res = norm.forward(&input);
        assert!(res.is_err());
    }

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
