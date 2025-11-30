use crate::nn::Linear;
use crate::tensor::{Cpu, Result, Tensor, TensorElem, TensorOps};

/// LoRA (Low-Rank Adaptation) Linear Layer.
///
/// Wraps a frozen `Linear` layer and adds a low-rank adapter path:
/// `y = x @ W.T + x @ A.T @ B.T * scaling`
/// where:
/// - `W` is the frozen pretrained weight.
/// - `A` is `[r, in_features]` (down-projection).
/// - `B` is `[out_features, r]` (up-projection).
///
/// Note: In standard LoRA papers, A is usually random Gaussian, B is zero.
///
/// # Mathematical Details
///
/// The forward pass computes:
/// $$ h = W_0 x + \Delta W x = W_0 x + B A x $$
///
/// - $W_0$ is the frozen weight matrix.
/// - $\Delta W = B A$ is the low-rank update.
/// - $A$ projects the input down to rank $r$.
/// - $B$ projects it back up to the output dimension.
/// - `scaling` is usually $\alpha / r$, used to reduce the need for hyperparameter tuning when $r$ changes.
#[derive(Debug)]
pub struct LoraLinear<T: TensorElem> {
    pub linear: Linear<T>, // Frozen
    pub lora_a: Tensor<T, 2, Cpu>,
    pub lora_b: Tensor<T, 2, Cpu>,
    pub scaling: T,
    pub r: usize,
}

impl<T: TensorElem> LoraLinear<T> {
    pub fn new(
        linear: Linear<T>,
        r: usize,
        alpha: T,
        lora_a: Tensor<T, 2, Cpu>,
        lora_b: Tensor<T, 2, Cpu>,
    ) -> Self {
        // scaling = alpha / r
        // We can't compute division for generic T easily without Num traits or similar.
        // Assuming T implements Div.
        // For now, let's just take scaling as input or assume alpha=r (scaling=1).
        // Let's assume alpha is passed as scaling factor directly for simplicity.
        Self {
            linear,
            lora_a,
            lora_b,
            scaling: alpha,
            r,
        }
    }

    pub fn forward(&self, x: &Tensor<T, 2, Cpu>) -> Result<Tensor<T, 2, Cpu>> {
        // 1. Frozen path
        let frozen_out = self.linear.forward(x)?;

        // 2. LoRA path
        // x: [B, In]
        // A: [R, In] -> A.T: [In, R]
        // B: [Out, R] -> B.T: [R, Out]

        // x @ A.T -> [B, R]
        let a_t = self.lora_a.transpose()?;
        let xa = x.matmul(&a_t)?;

        // xa @ B.T -> [B, Out]
        let b_t = self.lora_b.transpose()?;
        let lora_out = xa.matmul(&b_t)?;

        // Scale and Add
        // frozen + lora * scaling
        // We iterate and add.

        let mut out = frozen_out;
        let lora_data = lora_out.data();
        let out_data = out.data_mut();

        // Assuming shapes match (checked by matmul implicitly)
        for (o, l) in out_data.iter_mut().zip(lora_data.iter()) {
            *o = o.add(l.mul(self.scaling));
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_lora_linear_new() {
        let linear = Linear::new(Tensor::zeros([4, 2]), None);
        let lora_a = Tensor::zeros([1, 2]);
        let lora_b = Tensor::zeros([4, 1]);
        let lora = LoraLinear::new(linear, 1, 1.0, lora_a, lora_b);

        assert_eq!(lora.lora_a.shape(), &[1, 2]); // [Rank, In]
        assert_eq!(lora.lora_b.shape(), &[4, 1]); // [Out, Rank]
    }

    #[test]
    fn test_lora_linear_forward() {
        // Base: Identity [2, 2]
        // LoRA A: Ones [1, 2]
        // LoRA B: Ones [2, 1]
        // Scaling: 1.0
        // Input: [1, 2] = [[1, 1]]

        // Base Path: [1, 1] @ I = [1, 1]
        // LoRA Path:
        // x @ A.T = [1, 1] @ [1; 1] = [2] (Rank 1)
        // [2] @ B.T = [2] @ [1, 1] = [2, 2]
        // Output = Base + LoRA = [1+2, 1+2] = [3, 3]

        let w = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], [2, 2]).unwrap();
        let linear = Linear::new(w, None);

        let lora_a = Tensor::ones([1, 2]);
        let lora_b = Tensor::ones([2, 1]);

        let lora = LoraLinear::new(linear, 1, 1.0, lora_a, lora_b);

        let x = Tensor::ones([1, 2]);
        let y = lora.forward(&x).unwrap();

        let data = y.data();
        assert!((data[0] - 3.0f64).abs() < 1e-6);
        assert!((data[1] - 3.0f64).abs() < 1e-6);
    }
}
