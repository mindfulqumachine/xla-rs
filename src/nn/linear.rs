use crate::tensor::{Cpu, Tensor, TensorElem, TensorOps};

use rayon::prelude::*;

/// Constants for Linear Layer
const WEIGHT_RANK: usize = 2;
const BIAS_RANK: usize = 1;

/// Trait to enforce allowed ranks for Linear layer forward pass.
pub trait AllowedLinearRank<const N: usize> {}
impl AllowedLinearRank<2> for () {}
impl AllowedLinearRank<3> for () {}

/// # Design Philosophy: Fixed Ranks
///
/// The `Linear` layer enforces `WEIGHT_RANK = 2` and `BIAS_RANK = 1`.
/// This is a deliberate design choice to align with the mathematical definition of a Linear (or Fully Connected) layer:
/// $$y = xA^T + b$$
///
/// - **Weights ($A$):** Must be a Matrix (Rank 2) mapping `in_features` $\to$ `out_features`.
/// - **Bias ($b$):** Must be a Vector (Rank 1) matching `out_features`.
///
/// While one might want to use a `Tensor<T, 16, Cpu>` as weights, doing so would strictly no longer be a
/// standard "Linear" layer operation (it would be a Tensor Contraction or specialized convolution).
/// By enforcing these ranks, we keep the `Linear` abstraction clean, predictable, and mathematically correct.
/// If higher-dimensional weights are needed, they should be explicitly reshaped or flattened before being
/// passed to a Linear layer.
///
/// Linear Layer: `y = xA^T + b`
///
/// Performs a linear transformation on the input data.
/// This layer represents a collection of neurons where every input is connected to every output.
///
/// # Generics
/// - `T`: The element type of the tensors (e.g., `f32`, `f64`).
///
/// # Examples
/// ```rust
/// use xla_rs::nn::Linear;
/// use xla_rs::tensor::Tensor;
/// // Create a layer with 10 inputs and 5 outputs
/// let layer = Linear::<f32>::new(
///     Tensor::zeros([5, 10]), // Weights: [out, in]
///     Some(Tensor::zeros([5])) // Bias: [out]
/// );
/// ```
#[derive(Debug)]
pub struct Linear<T: TensorElem> {
    /// The learnable weights of the layer.
    /// - Shape: `[out_features, in_features]`
    pub weight: Tensor<T, WEIGHT_RANK, Cpu>,

    /// The learnable bias of the layer.
    /// - Shape: `[out_features]`
    pub bias: Option<Tensor<T, BIAS_RANK, Cpu>>,
}

impl<T: TensorElem> Linear<T> {
    /// Creates a new Linear layer.
    ///
    /// # Arguments
    ///
    /// * `weight` - The weight tensor of shape `[out_features, in_features]`.
    /// * `bias` - The optional bias tensor of shape `[out_features]`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use xla_rs::nn::Linear;
    /// use xla_rs::tensor::Tensor;
    ///
    /// let weight = Tensor::<f32, 2, _>::zeros([5, 10]);
    /// let layer = Linear::new(weight, None);
    /// ```
    pub fn new(
        weight: Tensor<T, WEIGHT_RANK, Cpu>,
        bias: Option<Tensor<T, BIAS_RANK, Cpu>>,
    ) -> Self {
        Self { weight, bias }
    }

    /// Performs the forward pass of the Linear layer.
    ///
    /// Supports inputs of Rank 2 `[batch_size, in_features]` or Rank 3 `[batch_size, seq_len, in_features]`.
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying the linear transformation.
    pub fn forward<const RANK: usize>(
        &self,
        x: &Tensor<T, RANK, Cpu>,
    ) -> crate::tensor::Result<Tensor<T, RANK, Cpu>>
    where
        (): AllowedLinearRank<RANK>,
    {
        // Compile-time check (redundant with trait bound but satisfies request for safer check if desired,
        // but trait bound `AllowedLinearRank` actually prevents compilation for other ranks, so strictly unnecessary
        // to assert constant. However, we'll stick to the logic that handles 2 and 3).

        let w_t = self.weight.transpose()?;

        if RANK == 3 {
            let [b, s, i] = x.shape()[0..3].try_into().unwrap();
            // Explicitly specify type for flat_x
            let flat_x: Tensor<T, 2, Cpu> = x.clone().reshape([b * s, i])?;
            let out_flat = flat_x.matmul(&w_t)?;

            let out_biased = if let Some(b_bias) = &self.bias {
                Self::add_bias(&out_flat, b_bias)?
            } else {
                out_flat
            };

            let out_features = self.weight.shape()[0];
            let res = out_biased.reshape([b, s, out_features])?;

            unsafe {
                let res_ptr = &res as *const Tensor<T, 3, Cpu>;
                let ret = std::ptr::read(res_ptr as *const Tensor<T, RANK, Cpu>);
                std::mem::forget(res);
                Ok(ret)
            }
        } else {
            // RANK == 2 guaranteed by AllowedLinearRank<RANK> if not 3

            // We need to cast `w_t` (Rank 2) to `Tensor<T, RANK>` unsafely to call `matmul`
            // because compiler sees generic RANK.

            let w_t_cast: &Tensor<T, RANK, Cpu> = unsafe { std::mem::transmute(&w_t) };

            let out = x.matmul(w_t_cast)?;
            let out = if let Some(b) = &self.bias {
                let shape = out.shape();
                let [_rows, cols] = [shape[0], shape[1]];

                if b.shape()[0] != cols {
                    return Err(crate::tensor::TensorError::ShapeMismatch {
                        expected: vec![cols],
                        got: vec![b.shape()[0]],
                    });
                }

                let bias_data = b.data();
                let mut out_mut = out;
                let out_slice = out_mut.data_mut();

                out_slice.par_chunks_mut(cols).for_each(|row| {
                    for (r, bv) in row.iter_mut().zip(bias_data.iter()) {
                        *r += *bv;
                    }
                });
                out_mut
            } else {
                out
            };
            Ok(out)
        }
    }

    /// Helper to add bias to a 2D tensor.
    fn add_bias(
        x: &Tensor<T, 2, Cpu>,
        bias: &Tensor<T, 1, Cpu>,
    ) -> crate::tensor::Result<Tensor<T, 2>> {
        let [_, cols] = *x.shape();
        let [b_cols] = *bias.shape();

        if cols != b_cols {
            return Err(crate::tensor::TensorError::ShapeMismatch {
                expected: vec![cols],
                got: vec![b_cols],
            });
        }

        let mut out = x.clone();

        out.data_mut().par_chunks_mut(cols).for_each(|row| {
            for (r, b) in row.iter_mut().zip(bias.data().iter()) {
                *r += *b;
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
    fn test_linear_new() {
        let weight = Tensor::<f32, 2, Cpu>::zeros([5, 10]);
        let bias = Tensor::<f32, 1, Cpu>::zeros([5]);
        let layer = Linear::new(weight, Some(bias));
        assert!(layer.bias.is_some());
    }

    #[test]
    fn test_linear_forward_rank2() {
        // Input: [2, 3]
        // Weight: [4, 3] (out=4, in=3)
        // Bias: [4]
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::<f32, 2, Cpu>::new(input_data, [2, 3]).unwrap();

        let weight_data = vec![
            1.0, 0.0, 0.0, // 1st neuron
            0.0, 1.0, 0.0, // 2nd neuron
            0.0, 0.0, 1.0, // 3rd neuron
            1.0, 1.0, 1.0, // 4th neuron
        ];
        let weight = Tensor::<f32, 2, Cpu>::new(weight_data, [4, 3]).unwrap();

        let bias_data = vec![0.1, 0.2, 0.3, 0.4];
        let bias = Tensor::<f32, 1, Cpu>::new(bias_data, [4]).unwrap();

        let layer = Linear::new(weight, Some(bias));
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 4]);
        // Row 1: [1, 2, 3]
        // Out 1: 1*1 + 0.1 = 1.1
        // Out 2: 2*1 + 0.2 = 2.2
        // Out 3: 3*1 + 0.3 = 3.3
        // Out 4: (1+2+3) + 0.4 = 6.4
        let out_data = output.data();
        assert!((out_data[0] - 1.1).abs() < 1e-6);
        assert!((out_data[1] - 2.2).abs() < 1e-6);
        assert!((out_data[2] - 3.3).abs() < 1e-6);
        assert!((out_data[3] - 6.4).abs() < 1e-6);
    }

    #[test]
    fn test_linear_forward_rank3() {
        // Input: [1, 2, 3] (Batch=1, Seq=2, In=3)
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::<f32, 3, Cpu>::new(input_data, [1, 2, 3]).unwrap();

        let weight_data = vec![
            1.0, 1.0, 1.0, // 1st neuron
            2.0, 2.0, 2.0, // 2nd neuron
        ];
        let weight = Tensor::<f32, 2, Cpu>::new(weight_data, [2, 3]).unwrap();

        let layer = Linear::new(weight, None);
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 2, 2]);
        // Row 1: [1, 2, 3] -> Sum=6. Out1=6, Out2=12
        // Row 2: [4, 5, 6] -> Sum=15. Out1=15, Out2=30
        let out_data = output.data();
        assert!((out_data[0] - 6.0).abs() < 1e-6);
        assert!((out_data[1] - 12.0).abs() < 1e-6);
        assert!((out_data[2] - 15.0).abs() < 1e-6);
        assert!((out_data[3] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_shape_mismatch() {
        let weight = Tensor::<f32, 2, Cpu>::zeros([5, 10]);
        let bias = Tensor::<f32, 1, Cpu>::zeros([4]); // Wrong size
        let layer = Linear::new(weight, Some(bias));

        let input = Tensor::<f32, 2, Cpu>::zeros([2, 10]);
        // This should fail inside forward when adding bias, or ideally we check in new?
        // Current impl checks in add_bias which is called in forward.
        // Actually, let's check add_bias directly via private access if possible or just run forward.
        // Since we are in the same module (submodule tests), we can test private methods if we want,
        // but forward is public.

        let res = layer.forward(&input);
        assert!(res.is_err());
    }

    #[test]
    fn test_linear_forward_rank2_no_bias() {
        let input_data = vec![1.0, 2.0];
        let input = Tensor::<f32, 2, Cpu>::new(input_data, [1, 2]).unwrap();

        // Weight: [2, 2]
        let weight_data = vec![1.0, 0.0, 0.0, 1.0];
        let weight = Tensor::<f32, 2, Cpu>::new(weight_data, [2, 2]).unwrap();

        let layer = Linear::new(weight, None);
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 2]);
        let out_data = output.data();
        assert!((out_data[0] - 1.0).abs() < 1e-6);
        assert!((out_data[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_forward_rank3_with_bias() {
        // Input: [1, 2, 2] (Batch=1, Seq=2, In=2)
        let input_data = vec![1.0, 1.0, 2.0, 2.0];
        let input = Tensor::<f32, 3, Cpu>::new(input_data, [1, 2, 2]).unwrap();

        // Weight: [2, 2] (Identity)
        let weight_data = vec![1.0, 0.0, 0.0, 1.0];
        let weight = Tensor::<f32, 2, Cpu>::new(weight_data, [2, 2]).unwrap();

        // Bias: [2]
        let bias_data = vec![0.5, 0.5];
        let bias = Tensor::<f32, 1, Cpu>::new(bias_data, [2]).unwrap();

        let layer = Linear::new(weight, Some(bias));
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 2, 2]);
        let out_data = output.data();
        // Row 1: [1, 1] + [0.5, 0.5] = [1.5, 1.5]
        // Row 2: [2, 2] + [0.5, 0.5] = [2.5, 2.5]
        assert!((out_data[0] - 1.5).abs() < 1e-6);
        assert!((out_data[1] - 1.5).abs() < 1e-6);
        assert!((out_data[2] - 2.5).abs() < 1e-6);
        assert!((out_data[3] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_linear_forward_rank3_mismatch() {
        let weight = Tensor::<f32, 2, Cpu>::zeros([2, 2]);
        let bias = Tensor::<f32, 1, Cpu>::zeros([3]); // Wrong size
        let layer = Linear::new(weight, Some(bias));

        let input = Tensor::<f32, 3, Cpu>::zeros([1, 2, 2]);
        let res = layer.forward(&input);
        assert!(res.is_err());
    }
}
