use crate::tensor::{Tensor, TensorElem, Device, Cpu};
use std::rc::Rc;
use std::cell::RefCell;

/// A Variable wraps a Tensor and tracks its gradient and computational history.
///
/// For simplicity, we use Rc<RefCell> for shared state in the graph (single-threaded training for now).
#[derive(Clone, Debug)]
pub struct Variable<T, const RANK: usize, D: Device = Cpu>
where T: TensorElem {
    pub data: Tensor<T, RANK, D>,
    pub grad: Option<Tensor<T, RANK, D>>,
    pub requires_grad: bool,
    // Graph history would go here (e.g., `grad_fn`)
}

impl<T, const RANK: usize, D: Device> Variable<T, RANK, D>
where T: TensorElem {
    pub fn new(data: Tensor<T, RANK, D>, requires_grad: bool) -> Self {
        Self {
            data,
            grad: None,
            requires_grad,
        }
    }

    pub fn backward(&mut self) {
        if !self.requires_grad {
            return;
        }
        // TODO: Implement backward pass using topological sort of grad_fn
    }

    pub fn zero_grad(&mut self) {
        self.grad = None;
    }
}
