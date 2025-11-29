//! Automatic Differentiation (Autograd) module.
//!
//! This module implements a "Define-by-Run" (Tape-based) automatic differentiation system,
//! similar to PyTorch. It allows for automatic calculation of gradients for tensor operations,
//! which is essential for training neural networks.
//!
//! # Key Components
//!
//! - [`Variable`]: The core struct that wraps a `Tensor` and tracks its gradient and computation history.
//! - [`GraphNode`]: A trait representing an operation in the computation graph.
//! - [`engine::backward`]: The engine that performs the backward pass (topological sort and gradient propagation).
//! - [`functional`]: A submodule providing a JAX-style functional API (`grad`, `value_and_grad`).
//!
//! # Example
//!
//! ```rust
//! # use xla_rs::tensor::Tensor;
//! # use xla_rs::autograd::Variable;
//! let a = Variable::new(Tensor::new(vec![2.0], []).unwrap());
//! let b = Variable::new(Tensor::new(vec![3.0], []).unwrap());
//!
//! // c = a * b
//! let c = a.clone() * b.clone();
//!
//! c.backward();
//!
//! // dc/da = b = 3.0
//! assert_eq!(a.grad.borrow().as_ref().unwrap().data()[0], 3.0);
//! ```

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
