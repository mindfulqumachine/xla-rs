use super::{GraphNode, Variable};
use crate::tensor::{Cpu, Tensor, TensorElem};
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Add, Mul};
use std::rc::Rc;

// --- Add Node ---
/// A node representing element-wise addition in the computation graph.
#[derive(Debug)]
struct AddNode<T: TensorElem, const RANK: usize> {
    lhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    rhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    out_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    parents: Vec<Rc<dyn GraphNode>>,
}

impl<T: TensorElem, const RANK: usize> GraphNode for AddNode<T, RANK> {
    fn backward(&self) {
        if let Some(grad) = self.out_grad.borrow().as_ref() {
            // d(x+y)/dx = 1 * grad
            // d(x+y)/dy = 1 * grad

            // Accumulate gradient for lhs
            {
                let mut lhs = self.lhs_grad.borrow_mut();
                if let Some(l) = lhs.as_mut() {
                    *l = (l.add(grad)).unwrap();
                } else {
                    *lhs = Some(grad.clone());
                }
            }

            // Accumulate gradient for rhs
            {
                let mut rhs = self.rhs_grad.borrow_mut();
                if let Some(r) = rhs.as_mut() {
                    *r = (r.add(grad)).unwrap();
                } else {
                    *rhs = Some(grad.clone());
                }
            }
        }
    }

    fn parents(&self) -> Vec<Rc<dyn GraphNode>> {
        self.parents.clone()
    }
}

impl<T: TensorElem + 'static, const RANK: usize> Add for Variable<T, RANK> {
    type Output = Variable<T, RANK>;

    /// Adds two variables element-wise.
    ///
    /// This operation creates a new node in the computation graph.
    fn add(self, rhs: Self) -> Self::Output {
        let data = (&self.data + &rhs.data).unwrap();

        let mut parents = Vec::new();
        if let Some(p) = &self.node {
            parents.push(p.clone());
        }
        if let Some(p) = &rhs.node {
            parents.push(p.clone());
        }

        // Even leaf nodes need to be part of the graph if we want to backprop to them?
        // Actually, leaf variables usually don't have a `node` (creator).
        // But the `AddNode` needs to update their `grad`.
        // So `AddNode` holds references to their `grad` cells.

        let out_grad = Rc::new(RefCell::new(None));

        let node = Rc::new(AddNode {
            lhs_grad: self.grad.clone(),
            rhs_grad: rhs.grad.clone(),
            out_grad: out_grad.clone(),
            parents, // This is wrong. Parents should be the nodes that created lhs/rhs.
                     // If lhs is leaf, it has no parent node.
                     // But topological sort needs to traverse.
                     // If leaf has no node, traversal stops there. Correct.
        });

        Variable {
            data,
            grad: out_grad,
            node: Some(node),
        }
    }
}

// --- Mul Node ---
/// A node representing element-wise multiplication in the computation graph.
#[derive(Debug)]
struct MulNode<T: TensorElem, const RANK: usize> {
    lhs_data: Tensor<T, RANK, Cpu>,
    rhs_data: Tensor<T, RANK, Cpu>,
    lhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    rhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    out_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    parents: Vec<Rc<dyn GraphNode>>,
}

impl<T: TensorElem, const RANK: usize> GraphNode for MulNode<T, RANK> {
    fn backward(&self) {
        if let Some(grad) = self.out_grad.borrow().as_ref() {
            // d(x*y)/dx = y * grad
            // d(x*y)/dy = x * grad

            {
                let mut lhs = self.lhs_grad.borrow_mut();
                let dl_dx = (&self.rhs_data * grad).unwrap();
                if let Some(l) = lhs.as_mut() {
                    *l = (l.add(&dl_dx)).unwrap();
                } else {
                    *lhs = Some(dl_dx);
                }
            }

            {
                let mut rhs = self.rhs_grad.borrow_mut();
                let dr_dy = (&self.lhs_data * grad).unwrap();
                if let Some(r) = rhs.as_mut() {
                    *r = (r.add(&dr_dy)).unwrap();
                } else {
                    *rhs = Some(dr_dy);
                }
            }
        }
    }

    fn parents(&self) -> Vec<Rc<dyn GraphNode>> {
        self.parents.clone()
    }
}

impl<T: TensorElem + 'static, const RANK: usize> Mul for Variable<T, RANK> {
    type Output = Variable<T, RANK>;

    /// Multiplies two variables element-wise.
    ///
    /// This operation creates a new node in the computation graph.
    fn mul(self, rhs: Self) -> Self::Output {
        let data = (&self.data * &rhs.data).unwrap();

        let mut parents = Vec::new();
        if let Some(p) = &self.node {
            parents.push(p.clone());
        }
        if let Some(p) = &rhs.node {
            parents.push(p.clone());
        }

        let out_grad = Rc::new(RefCell::new(None));

        let node = Rc::new(MulNode {
            lhs_data: self.data.clone(),
            rhs_data: rhs.data.clone(),
            lhs_grad: self.grad.clone(),
            rhs_grad: rhs.grad.clone(),
            out_grad: out_grad.clone(),
            parents,
        });

        Variable {
            data,
            grad: out_grad,
            node: Some(node),
        }
    }
}

// --- MatMul Node ---
/// A node representing matrix multiplication in the computation graph.
#[derive(Debug)]
struct MatMulNode<T: TensorElem, const RANK: usize> {
    lhs_data: Tensor<T, RANK, Cpu>,
    rhs_data: Tensor<T, RANK, Cpu>,
    lhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    rhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    out_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    parents: Vec<Rc<dyn GraphNode>>,
}

impl<T: TensorElem, const RANK: usize> GraphNode for MatMulNode<T, RANK> {
    #[allow(clippy::collapsible_if)]
    fn backward(&self) {
        if let Some(grad) = self.out_grad.borrow().as_ref() {
            // LHS Gradient
            {
                let mut lhs = self.lhs_grad.borrow_mut();
                if let Ok(rhs_t) = self.rhs_data.transpose() {
                    if let Ok(dl_da) = grad.matmul(&rhs_t) {
                        if let Some(l) = lhs.as_mut() {
                            *l = (l.add(&dl_da)).unwrap();
                        } else {
                            *lhs = Some(dl_da);
                        }
                    }
                }
            }

            // RHS Gradient
            {
                let mut rhs = self.rhs_grad.borrow_mut();
                if let Ok(lhs_t) = self.lhs_data.transpose() {
                    if let Ok(dr_db) = lhs_t.matmul(grad) {
                        if let Some(r) = rhs.as_mut() {
                            *r = (r.add(&dr_db)).unwrap();
                        } else {
                            *rhs = Some(dr_db);
                        }
                    }
                }
            }
        }
    }

    fn parents(&self) -> Vec<Rc<dyn GraphNode>> {
        self.parents.clone()
    }
}

impl<T: TensorElem + 'static, const RANK: usize> Variable<T, RANK> {
    /// Performs matrix multiplication between two variables.
    ///
    /// This operation creates a new node in the computation graph.
    pub fn matmul(&self, rhs: &Self) -> crate::tensor::Result<Self> {
        let data = self.data.matmul(&rhs.data)?;

        let mut parents = Vec::new();
        if let Some(p) = &self.node {
            parents.push(p.clone());
        }
        if let Some(p) = &rhs.node {
            parents.push(p.clone());
        }

        let out_grad = Rc::new(RefCell::new(None));

        let node = Rc::new(MatMulNode {
            lhs_data: self.data.clone(),
            rhs_data: rhs.data.clone(),
            lhs_grad: self.grad.clone(),
            rhs_grad: rhs.grad.clone(),
            out_grad: out_grad.clone(),
            parents,
        });

        Ok(Variable {
            data,
            grad: out_grad,
            node: Some(node),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_backward() {
        let a = Variable::new(Tensor::new(vec![2.0], []).unwrap());
        let b = Variable::new(Tensor::new(vec![3.0], []).unwrap());
        let c = a.clone() + b.clone();

        c.backward();

        assert_eq!(a.grad.borrow().as_ref().unwrap().data()[0], 1.0);
        assert_eq!(b.grad.borrow().as_ref().unwrap().data()[0], 1.0);
    }

    #[test]
    fn test_mul_backward() {
        let a = Variable::new(Tensor::new(vec![2.0], []).unwrap());
        let b = Variable::new(Tensor::new(vec![3.0], []).unwrap());
        let c = a.clone() * b.clone();

        c.backward();

        assert_eq!(a.grad.borrow().as_ref().unwrap().data()[0], 3.0);
        assert_eq!(b.grad.borrow().as_ref().unwrap().data()[0], 2.0);
    }

    #[test]
    fn test_chain_rule() {
        // y = (a + b) * c
        // a=2, b=3, c=4
        // y = (2+3)*4 = 20
        // dy/da = c = 4
        // dy/db = c = 4
        // dy/dc = a + b = 5

        let a = Variable::new(Tensor::new(vec![2.0], []).unwrap());
        let b = Variable::new(Tensor::new(vec![3.0], []).unwrap());
        let c = Variable::new(Tensor::new(vec![4.0], []).unwrap());

        let sum = a.clone() + b.clone();
        let y = sum * c.clone();

        y.backward();

        assert_eq!(a.grad.borrow().as_ref().unwrap().data()[0], 4.0);
        assert_eq!(b.grad.borrow().as_ref().unwrap().data()[0], 4.0);
        assert_eq!(c.grad.borrow().as_ref().unwrap().data()[0], 5.0);
    }
}
