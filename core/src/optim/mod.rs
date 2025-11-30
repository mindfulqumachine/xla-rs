pub mod adamw;
pub mod sgd;
pub use adamw::AdamW;
pub use sgd::Sgd;

use crate::tensor::{Result, Tensor, TensorElem};

/// A trait for optimizers (e.g., SGD, Adam).
///
/// Optimizers are responsible for updating model parameters based on computed gradients.
pub trait Optimizer<T: TensorElem> {
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

    /// Updates a parameter tensor using its gradient.
    ///
    /// # Arguments
    /// * `key` - A unique identifier for the parameter (e.g., its index in the parameter list).
    ///   This is used by stateful optimizers (like Adam) to track moments.
    /// * `param` - The parameter tensor to update.
    /// * `grad` - The gradient tensor.
    fn update<const RANK: usize>(
        &mut self,
        key: usize,
        param: &mut Tensor<T, RANK, crate::tensor::Cpu>,
        grad: &Tensor<T, RANK, crate::tensor::Cpu>,
    ) -> Result<()>;
}
