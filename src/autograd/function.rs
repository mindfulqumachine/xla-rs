use crate::tensor::{Tensor, TensorElem, Device, Cpu};
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::Debug;

/// Trait for backward propagation functions.
pub trait Backward<T: TensorElem, const RANK: usize, D: Device>: Debug {
    /// Computes gradients for inputs given the gradient of the output.
    /// Returns a list of gradients corresponding to the inputs.
    fn apply(&self, grad_output: Tensor<T, RANK, D>) -> Vec<Option<Tensor<T, RANK, D>>>;
}

/// Represents a node in the computation graph (a record of an operation).
#[derive(Debug, Clone)]
pub struct Edge<T: TensorElem, const RANK: usize, D: Device> {
    // The backward function associated with the operation that produced a variable
    pub function: Rc<RefCell<dyn Backward<T, RANK, D>>>,
    // Index of the input in the backward function's output list (not used much in simple cases, but needed for multi-output)
    pub input_index: usize,
}

// However, we need to link Variables to this.
// A Variable created by an op has a `grad_fn`.
// When we run backward on `grad_fn`, we get grads for its inputs.
// We need to accumulate those grads into the input Variables.
// So `Backward` needs to know where to send the grads.
// Usually `Backward` function stores `next_edges` which are links to the `grad_fn` of the inputs, OR links to the AccumulateGrad nodes of inputs.

// Simplified design:
// `Variable` has `grad_fn`.
// `grad_fn` (struct AddBackward) stores `parents: Vec<Variable>`.
// `apply` computes grads.
// The engine pushes grads to parents.

// But wait, generic ownership cycle if Variable owns grad_fn and grad_fn owns Variable (parents).
// Use `Weak` for parents? Or `Variable` handles are just lightweight wrappers around a shared internal state.

// Let's try the "Variable holds data and grad, and separate Graph Node" approach.
// OR: PyTorch style. Variable *is* the handle.
// If we keep it simple:
// We only need backward for training.
// We can construct the graph dynamically.

// Let's define `Context` to save tensors.
