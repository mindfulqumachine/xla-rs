use crate::tensor::{Tensor, TensorElem, Device, Cpu};
use crate::autograd::variable::Variable;
use crate::autograd::variable::Backward;
use std::ops::{Add, Mul};
use std::rc::Rc;
use std::cell::RefCell;

// --- Add ---

#[derive(Debug)]
pub struct AddBackward<T: TensorElem, const RANK: usize> {
    lhs: Variable<T, RANK, Cpu>,
    rhs: Variable<T, RANK, Cpu>,
}

impl<T: TensorElem, const RANK: usize> Backward<T, RANK, Cpu> for AddBackward<T, RANK> {
    fn apply(&self, grad_output: Tensor<T, RANK, Cpu>) -> Vec<Option<Tensor<T, RANK, Cpu>>> {
        // d(a+b)/da = 1 * grad_output
        // d(a+b)/db = 1 * grad_output
        vec![Some(grad_output.clone()), Some(grad_output)]
    }

    fn next_variables(&self) -> Vec<Variable<T, RANK, Cpu>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }
}

impl<T: TensorElem, const RANK: usize> Add for &Variable<T, RANK, Cpu> {
    type Output = Variable<T, RANK, Cpu>;

    fn add(self, rhs: Self) -> Self::Output {
        let data = (&self.data() + &rhs.data()).expect("Variable Add Failed");

        let requires_grad = self.inner.borrow().requires_grad || rhs.inner.borrow().requires_grad;

        if requires_grad {
            let grad_fn = Rc::new(RefCell::new(AddBackward {
                lhs: self.clone(),
                rhs: rhs.clone(),
            }));
            Variable::with_grad_fn(data, true, grad_fn)
        } else {
            Variable::new(data, false)
        }
    }
}

// --- Matmul ---

#[derive(Debug)]
pub struct MatmulBackward<T: TensorElem, const RANK: usize> {
    lhs: Variable<T, RANK, Cpu>,
    rhs: Variable<T, RANK, Cpu>,
}

impl<T: TensorElem, const RANK: usize> Backward<T, RANK, Cpu> for MatmulBackward<T, RANK> {
    fn apply(&self, grad_output: Tensor<T, RANK, Cpu>) -> Vec<Option<Tensor<T, RANK, Cpu>>> {
        // C = A @ B
        // dA = dC @ B.T
        // dB = A.T @ dC

        let a = self.lhs.data();
        let b = self.rhs.data();

        let grad_a = if self.lhs.inner.borrow().requires_grad {
            let b_t = b.transpose().expect("Transpose failed");
            Some(grad_output.matmul(&b_t).expect("Grad A Matmul failed"))
        } else {
            None
        };

        let grad_b = if self.rhs.inner.borrow().requires_grad {
            let a_t = a.transpose().expect("Transpose failed");
            Some(a_t.matmul(&grad_output).expect("Grad B Matmul failed"))
        } else {
            None
        };

        vec![grad_a, grad_b]
    }

    fn next_variables(&self) -> Vec<Variable<T, RANK, Cpu>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }
}

impl<T: TensorElem, const RANK: usize> Variable<T, RANK, Cpu> {
    pub fn matmul(&self, rhs: &Self) -> Self {
        let data = self.data().matmul(&rhs.data()).expect("Variable Matmul Failed");

        let requires_grad = self.inner.borrow().requires_grad || rhs.inner.borrow().requires_grad;

        if requires_grad {
            let grad_fn = Rc::new(RefCell::new(MatmulBackward {
                lhs: self.clone(),
                rhs: rhs.clone(),
            }));
            Variable::with_grad_fn(data, true, grad_fn)
        } else {
            Variable::new(data, false)
        }
    }
}
