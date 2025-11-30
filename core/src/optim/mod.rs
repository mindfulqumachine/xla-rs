pub mod sgd;
pub use sgd::Sgd;

use crate::tensor::{Result, Tensor, TensorElem};

/// A trait for optimizers (e.g., SGD, Adam).
///
/// Optimizers are responsible for updating model parameters based on computed gradients.
pub trait Optimizer<T: TensorElem> {
    /// Performs a single optimization step.
    ///
    /// # Arguments
    ///
    /// * `params` - A list of mutable references to the model parameters (weights/biases).
    /// * `grads` - A list of references to the gradients corresponding to the parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of parameters and gradients do not match,
    /// or if shapes mismatch.
    fn step(
        &mut self,
        params: &mut [&mut Tensor<T, 2, crate::tensor::Cpu>],
        grads: &[&Tensor<T, 2, crate::tensor::Cpu>],
    ) -> Result<()>;

    // Note: The above signature is a bit restrictive (Rank 2).
    // Realistically, params can have any rank.
    // We might need a generic way to handle `Tensor<T, RANK, Cpu>`.
    // But Rust traits don't support arbitrary rank generics easily in a list.
    // Option A: Use `Tensor<T, 1, Cpu>` (flattened) for everything?
    // Option B: Use `Box<dyn AnyTensor>`?
    // Option C: Just support Rank 1 and 2 for now, or specific ranks.
    //
    // Given our `Linear` uses Rank 2 (weight) and Rank 1 (bias), and `Embedding` uses Rank 2.
    // `RMSNorm` uses Rank 1.
    //
    // Let's try to make `step` generic or take a trait object if possible, but `Tensor` is a struct.
    //
    // Alternative: `Optimizer` struct holds references to params? No, params change.
    //
    // Let's define `step` to take a closure or iterator?
    //
    // Simpler approach for this MVP:
    // `step` takes `&mut Tensor` and `&Tensor` (grad).
    // But usually we pass all params.
    //
    // Let's define `step_tensor` which updates a single tensor.
    // And `step` which iterates?
    //
    // Let's stick to a simple `step` that takes `&mut Tensor` and `&Tensor` for a single parameter update.
    // The training loop will iterate over params.

    fn update<const RANK: usize>(
        &self,
        param: &mut Tensor<T, RANK, crate::tensor::Cpu>,
        grad: &Tensor<T, RANK, crate::tensor::Cpu>,
    ) -> Result<()>;
}
