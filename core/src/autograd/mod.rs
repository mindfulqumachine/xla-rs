//! Automatic Differentiation (Autograd) module.
//!
//! # What is Autograd?
//!
//! Automatic Differentiation (AD) is a technique to evaluate the derivative of a function specified
//! by a computer program. It is the backbone of modern deep learning, allowing us to compute
//! gradients of the loss function with respect to model parameters for backpropagation.
//!
//! `xla-rs` implements **Reverse-Mode AD** (also known as backpropagation) using a **Tape-based** approach.
//!
//! # How it Works
//!
//! 1. **Forward Pass**: When you perform operations on `Variable`s (wrappers around Tensors),
//!    `xla-rs` builds a computation graph (a DAG) dynamically. Each node in the graph represents
//!    an operation (e.g., addition, multiplication).
//! 2. **Backward Pass**: When you call `.backward()` on a scalar variable (usually the loss),
//!    the engine traverses the graph in reverse topological order, computing gradients for each
//!    node using the chain rule.
//!
//! # Example: Simple Gradient Computation
//!
//! We want to compute the derivative of $f(x) = x^2$ at $x = 3$.
//! $f'(x) = 2x$, so $f'(3) = 6$.
//!
//! ```rust
//! use xla_rs::tensor::Tensor;
//! use xla_rs::autograd::Variable;
//!
//! // 1. Define input variable (requires gradient)
//! let data = Tensor::new(vec![3.0], []).unwrap();
//! let x = Variable::new(data);
//!
//! // 2. Perform operation: y = x * x
//! let y = x.clone() * x.clone();
//!
//! // 3. Backward pass
//! y.backward();
//!
//! // 4. Check gradient: dy/dx = 2 * x = 6.0
//! let grad = x.grad.borrow();
//! assert_eq!(grad.as_ref().unwrap().data()[0], 6.0);
//! ```
//!
//! > [!TIP]
//! > **Expert Note: Wengert List (Tape)**
//! > This implementation uses a "Define-by-Run" scheme. The "tape" is implicitly formed by the
//! > `Rc<dyn GraphNode>` links between variables. This is flexible (allows control flow like loops)
//! > but can be memory-intensive as the graph grows. Production frameworks often use graph optimization
//! > passes to reduce memory footprint.

use crate::tensor::{Cpu, Tensor, TensorElem};
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

pub mod engine;
pub mod functional;
pub mod ops;

/// A node in the computation graph.
///
/// This trait represents an operation that can be backpropagated through.
pub trait GraphNode: Debug {
    /// Computes the gradient for this node and propagates it to its parents.
    fn backward(&self);
    /// Returns the parent nodes of this node.
    fn parents(&self) -> Vec<Rc<dyn GraphNode>>;
}

/// A variable in the computation graph.
///
/// Wraps a `Tensor` and tracks its gradient and the operation that created it.
#[derive(Clone, Debug)]
pub struct Variable<T, const RANK: usize>
where
    T: TensorElem,
{
    /// The actual tensor data.
    pub data: Tensor<T, RANK, Cpu>,
    /// The gradient of the loss with respect to this variable.
    pub grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    /// The node in the computation graph that produced this variable.
    pub node: Option<Rc<dyn GraphNode>>,
}

impl<T, const RANK: usize> Variable<T, RANK>
where
    T: TensorElem + 'static,
{
    /// Creates a new leaf variable.
    ///
    /// Leaf variables are the inputs to the computation graph (e.g., weights, input data).
    /// They do not have a parent node.
    pub fn new(data: Tensor<T, RANK, Cpu>) -> Self {
        Self {
            data,
            grad: Rc::new(RefCell::new(None)),
            node: None,
        }
    }

    /// Creates a new variable with an associated graph node.
    ///
    /// This is typically used internally by operations to create output variables.
    pub fn with_node(data: Tensor<T, RANK, Cpu>, node: Rc<dyn GraphNode>) -> Self {
        Self {
            data,
            grad: Rc::new(RefCell::new(None)),
            node: Some(node),
        }
    }

    /// Triggers the backward pass starting from this variable.
    ///
    /// This variable is typically the loss value (a scalar).
    /// The gradient of this variable is seeded with 1.0.
    pub fn backward(&self) {
        // Seed gradient
        if self.grad.borrow().is_none() {
            *self.grad.borrow_mut() = Some(Tensor::ones(*self.data.shape()));
        }

        crate::autograd::engine::backward(self.node.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_creation() {
        let data = Tensor::new(vec![1.0, 2.0], [2]).unwrap();
        let var = Variable::new(data.clone());

        assert_eq!(var.data.data(), data.data());
        assert!(var.grad.borrow().is_none());
        assert!(var.node.is_none());
    }

    #[test]
    fn test_variable_backward_seed() {
        let data = Tensor::new(vec![1.0], []).unwrap();
        let var = Variable::new(data);

        // Backward on leaf node should just seed the gradient
        var.backward();

        assert!(var.grad.borrow().is_some());
        assert_eq!(var.grad.borrow().as_ref().unwrap().data()[0], 1.0);
    }

    #[test]
    fn test_variable_with_node() {
        let data = Tensor::new(vec![10.0], []).unwrap();

        // Create a mock node (using a simple struct that implements GraphNode)
        #[derive(Debug)]
        struct MockNode;
        impl GraphNode for MockNode {
            fn backward(&self) {}
            fn parents(&self) -> Vec<Rc<dyn GraphNode>> {
                vec![]
            }
        }

        let node = Rc::new(MockNode);
        let var = Variable::with_node(data.clone(), node.clone());

        assert_eq!(var.data.data(), data.data());
        assert!(var.node.is_some());
        assert!(var.grad.borrow().is_none());
    }
}
