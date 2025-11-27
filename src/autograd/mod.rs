use crate::tensor::{Cpu, Tensor, TensorElem};
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

pub mod engine;
pub mod ops;

pub trait GraphNode: Debug {
    fn backward(&self);
    fn parents(&self) -> Vec<Rc<dyn GraphNode>>;
}

#[derive(Clone, Debug)]
pub struct Variable<T, const RANK: usize>
where
    T: TensorElem,
{
    pub data: Tensor<T, RANK, Cpu>,
    pub grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    pub node: Option<Rc<dyn GraphNode>>,
}

impl<T, const RANK: usize> Variable<T, RANK>
where
    T: TensorElem + 'static,
{
    pub fn new(data: Tensor<T, RANK, Cpu>) -> Self {
        Self {
            data,
            grad: Rc::new(RefCell::new(None)),
            node: None,
        }
    }

    pub fn with_node(data: Tensor<T, RANK, Cpu>, node: Rc<dyn GraphNode>) -> Self {
        Self {
            data,
            grad: Rc::new(RefCell::new(None)),
            node: Some(node),
        }
    }

    pub fn backward(&self) {
        // Seed gradient
        if self.grad.borrow().is_none() {
            *self.grad.borrow_mut() = Some(Tensor::ones(*self.data.shape()));
        }

        crate::autograd::engine::backward(self.node.clone());
    }
}
