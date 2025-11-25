use crate::tensor::{Tensor, TensorElem, Device, Cpu};
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::Debug;

pub trait Backward<T: TensorElem, const RANK: usize, D: Device>: Debug {
    fn apply(&self, grad_output: Tensor<T, RANK, D>) -> Vec<Option<Tensor<T, RANK, D>>>;
    fn next_variables(&self) -> Vec<Variable<T, RANK, D>>;
}

#[derive(Clone, Debug)]
pub struct Variable<T, const RANK: usize, D: Device = Cpu>
where T: TensorElem {
    pub inner: Rc<RefCell<VariableInner<T, RANK, D>>>,
}

#[derive(Debug)]
pub struct VariableInner<T, const RANK: usize, D: Device>
where T: TensorElem {
    pub data: Tensor<T, RANK, D>,
    pub grad: Option<Tensor<T, RANK, D>>,
    pub requires_grad: bool,
    pub grad_fn: Option<Rc<RefCell<dyn Backward<T, RANK, D>>>>,
    pub is_leaf: bool,
}

// Generic impl for structural methods
impl<T, const RANK: usize, D: Device> Variable<T, RANK, D>
where T: TensorElem {
    pub fn new(data: Tensor<T, RANK, D>, requires_grad: bool) -> Self {
        Self {
            inner: Rc::new(RefCell::new(VariableInner {
                data,
                grad: None,
                requires_grad,
                grad_fn: None,
                is_leaf: true,
            }))
        }
    }

    pub fn with_grad_fn(
        data: Tensor<T, RANK, D>,
        requires_grad: bool,
        grad_fn: Rc<RefCell<dyn Backward<T, RANK, D>>>
    ) -> Self {
        Self {
            inner: Rc::new(RefCell::new(VariableInner {
                data,
                grad: None,
                requires_grad,
                grad_fn: Some(grad_fn),
                is_leaf: false,
            }))
        }
    }

    pub fn data(&self) -> Tensor<T, RANK, D> {
        self.inner.borrow().data.clone()
    }

    pub fn grad(&self) -> Option<Tensor<T, RANK, D>> {
        self.inner.borrow().grad.clone()
    }

    fn build_topo(&self, list: &mut Vec<Variable<T, RANK, D>>, visited: &mut std::collections::HashSet<usize>) {
        let ptr = self.inner.as_ptr() as usize;
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        let nexts = {
            let inner = self.inner.borrow();
            if let Some(gf) = &inner.grad_fn {
                gf.borrow().next_variables()
            } else {
                Vec::new()
            }
        };

        for parent in nexts {
            parent.build_topo(list, visited);
        }

        list.push(self.clone());
    }
}

// CPU specific implementation for Autograd logic (backward/add_grad)
impl<T, const RANK: usize> Variable<T, RANK, Cpu>
where T: TensorElem {

    pub fn add_grad(&self, g: Tensor<T, RANK, Cpu>) {
        let mut inner = self.inner.borrow_mut();
        if let Some(current) = &inner.grad {
            use std::ops::Add;
            // Assuming Tensor implements Add for &Tensor on Cpu (which it does via ops.rs)
            // We need to dereference or match types.
            let res = (current + &g).expect("Gradient accumulation shape mismatch");
            inner.grad = Some(res);
        } else {
            inner.grad = Some(g);
        }
    }

    pub fn backward(&self) {
        if !self.inner.borrow().requires_grad {
            return;
        }

        {
            let mut inner = self.inner.borrow_mut();
            if inner.grad.is_none() {
                inner.grad = Some(Tensor::ones(inner.data.shape().clone()));
            }
        }

        let mut topo_order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.build_topo(&mut topo_order, &mut visited);

        for var in topo_order.iter().rev() {
            var.fire_backward();
        }
    }

    fn fire_backward(&self) {
        let (grad_fn, grad) = {
            let inner = self.inner.borrow();
            if inner.grad_fn.is_none() || inner.grad.is_none() {
                return;
            }
            (inner.grad_fn.clone().unwrap(), inner.grad.clone().unwrap())
        };

        let grads = grad_fn.borrow().apply(grad);
        let nexts = grad_fn.borrow().next_variables();

        if grads.len() != nexts.len() {
            panic!("Backward function returned wrong number of gradients");
        }

        for (g_opt, next_var) in grads.into_iter().zip(nexts) {
            if let Some(g) = g_opt {
                next_var.add_grad(g);
            }
        }
    }
}
