use crate::tensor::{Cpu, Tensor, TensorElem};
use num_traits::Float;

/// Computes the SiLU (Sigmoid Linear Unit) activation function.
///
/// $$ \text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} $$
pub fn silu<T: TensorElem + Float>(x: T) -> T {
    let val = x.to_f32().unwrap();
    let sig = 1.0 / (1.0 + (-val).exp());
    T::from_f32(val * sig).unwrap()
}

/// Activation functions namespace.
pub struct Activation;

impl Activation {
    /// Applies the SiLU activation function element-wise to a tensor.
    pub fn silu<const RANK: usize, T: TensorElem + Float>(
        x: &Tensor<T, RANK, Cpu>,
    ) -> Tensor<T, RANK, Cpu> {
        x.map(silu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

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
