//! Activation Functions.
//!
//! # What is an Activation Function?
//!
//! Activation functions introduce **non-linearity** into neural networks. Without them, a neural network
//! (no matter how deep) would just be a single linear transformation (matrix multiplication).
//!
//! - **ReLU**: $f(x) = \max(0, x)$. Fast, simple, used in older models.
//! - **GELU**: Gaussian Error Linear Unit. Smoother than ReLU. Used in **GPT-2**, **BERT**.
//! - **SiLU** (Swish): Sigmoid Linear Unit. $x \cdot \sigma(x)$. Used in **Llama**, **Gemma**.

use crate::tensor::{Cpu, Tensor, TensorElem};
use num_traits::Float;

/// Computes the SiLU (Sigmoid Linear Unit) activation function.
///
/// $$ \text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} $$
///
/// Also known as "Swish". It is a smooth, non-monotonic function that consistently outperforms ReLU
/// on deep models.
pub fn silu<T: TensorElem + Float>(x: T) -> T {
    let val = x.to_f32().unwrap();
    let sig = 1.0 / (1.0 + (-val).exp());
    T::from_f32(val * sig).unwrap()
}

/// Activation functions namespace.
///
/// Provides element-wise activation functions for Tensors.
pub struct Activation;

impl Activation {
    /// Applies the SiLU activation function element-wise to a tensor.
    pub fn silu<const RANK: usize, T: TensorElem + Float>(
        x: &Tensor<T, RANK, Cpu>,
    ) -> Tensor<T, RANK, Cpu> {
        x.map(silu)
    }

    /// Applies the GELU activation function element-wise to a tensor.
    pub fn gelu<const RANK: usize, T: TensorElem + Float>(
        x: &Tensor<T, RANK, Cpu>,
    ) -> Tensor<T, RANK, Cpu> {
        x.map(gelu)
    }
}

/// Computes the GELU (Gaussian Error Linear Unit) activation function.
///
/// $$ \text{GELU}(x) \approx 0.5 x (1 + \tanh[\sqrt{2/\pi} (x + 0.044715 x^3)]) $$
///
/// This is the "tanh approximation" of GELU, which is faster to compute than the exact error function version.
/// It weights inputs by their magnitude, allowing small negative values to have a non-zero gradient (unlike ReLU).
pub fn gelu<T: TensorElem + Float>(x: T) -> T {
    let x_f = x.to_f32().unwrap();
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let c = 0.044715f32;

    let inner = sqrt_2_over_pi * (x_f + c * x_f.powi(3));
    let tanh_inner = inner.tanh();

    let res = 0.5 * x_f * (1.0 + tanh_inner);
    T::from_f32(res).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_gelu_value() {
        // gelu(0) = 0
        let val = 0.0f32;
        let res = gelu(val);
        assert!((res - 0.0).abs() < 1e-6);

        // gelu(1) approx 0.8413
        let val = 1.0f32;
        let res = gelu(val);
        // Manual calc:
        // sqrt(2/pi) = 0.79788
        // inner = 0.79788 * (1 + 0.044715) = 0.79788 * 1.044715 = 0.83355
        // tanh(0.83355) = 0.6824
        // 0.5 * 1 * (1 + 0.6824) = 0.5 * 1.6824 = 0.8412
        assert!((res - 0.8412).abs() < 1e-3);
    }

    #[test]
    fn test_activation_gelu_tensor() {
        let data = vec![0.0, 1.0];
        let tensor = Tensor::<f32, 1, Cpu>::new(data, [2]).unwrap();
        let res = Activation::gelu(&tensor);

        let res_data = res.data();
        assert!((res_data[0] - 0.0).abs() < 1e-6);
        assert!((res_data[1] - 0.8412).abs() < 1e-3);
    }

    #[test]
    fn test_silu_value() {
        let val = 2.0f32;
        let res = silu(val);
        // silu(2) = 2 * sigmoid(2) = 2 * (1 / (1 + exp(-2)))
        // exp(-2) approx 0.135335
        // 1 / 1.135335 approx 0.880797
        // 2 * 0.880797 approx 1.76159
        assert!((res - 1.76159).abs() < 1e-4);
    }

    #[test]
    fn test_activation_silu_tensor() {
        let data = vec![0.0, 2.0];
        let tensor = Tensor::<f32, 1, Cpu>::new(data, [2]).unwrap();
        let res = Activation::silu(&tensor);

        let res_data = res.data();
        assert!((res_data[0] - 0.0).abs() < 1e-6); // 0 * sigmoid(0) = 0
        assert!((res_data[1] - 1.76159).abs() < 1e-4);
    }
}
