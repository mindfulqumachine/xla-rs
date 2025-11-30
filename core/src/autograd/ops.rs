//! Operations for the autograd system.
//!
//! This module defines the nodes in the computation graph for various operations
//! (Add, Mul, MatMul) and implements the `backward` pass for each.

use super::{GraphNode, Variable};
use crate::tensor::{Cpu, Tensor, TensorElem, TensorOps};
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

// --- Add Node ---
/// A node representing element-wise addition in the computation graph.
// --- Add Node ---
/// A node representing element-wise addition in the computation graph.
#[derive(Debug)]
struct AddNode<T: TensorElem, const RANK: usize> {
    /// Gradient of the left-hand side operand.
    lhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    /// Gradient of the right-hand side operand.
    rhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    /// Gradient of the output (received from the parent node).
    out_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    /// Parent nodes in the computation graph.
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
// --- Sub Node ---
/// A node representing element-wise subtraction in the computation graph.
#[derive(Debug)]
struct SubNode<T: TensorElem, const RANK: usize> {
    lhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    rhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    out_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    parents: Vec<Rc<dyn GraphNode>>,
}

impl<T: TensorElem, const RANK: usize> GraphNode for SubNode<T, RANK> {
    fn backward(&self) {
        if let Some(grad) = self.out_grad.borrow().as_ref() {
            // d(x-y)/dx = 1 * grad
            // d(x-y)/dy = -1 * grad

            {
                let mut lhs = self.lhs_grad.borrow_mut();
                if let Some(l) = lhs.as_mut() {
                    *l = (l.add(grad)).unwrap();
                } else {
                    *lhs = Some(grad.clone());
                }
            }

            {
                let mut rhs = self.rhs_grad.borrow_mut();
                // grad * -1
                // We don't have neg() yet, so 0 - grad? Or grad * -1.
                // T::one().neg() might not exist for all T.
                // 0 - grad is safe.
                let zero = Tensor::zeros(*grad.shape());
                let neg_grad = (zero.sub(grad)).unwrap();

                if let Some(r) = rhs.as_mut() {
                    *r = (r.add(&neg_grad)).unwrap();
                } else {
                    *rhs = Some(neg_grad);
                }
            }
        }
    }

    fn parents(&self) -> Vec<Rc<dyn GraphNode>> {
        self.parents.clone()
    }
}

impl<T: TensorElem + 'static, const RANK: usize> Sub for Variable<T, RANK> {
    type Output = Variable<T, RANK>;

    fn sub(self, rhs: Self) -> Self::Output {
        let data = (&self.data - &rhs.data).unwrap();

        let mut parents = Vec::new();
        if let Some(p) = &self.node {
            parents.push(p.clone());
        }
        if let Some(p) = &rhs.node {
            parents.push(p.clone());
        }

        let out_grad = Rc::new(RefCell::new(None));

        let node = Rc::new(SubNode {
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

// --- Mul Node ---
/// A node representing element-wise multiplication in the computation graph.
// --- Mul Node ---
/// A node representing element-wise multiplication in the computation graph.
#[derive(Debug)]
struct MulNode<T: TensorElem, const RANK: usize> {
    /// Data of the left-hand side operand (needed for gradient calculation).
    lhs_data: Tensor<T, RANK, Cpu>,
    /// Data of the right-hand side operand (needed for gradient calculation).
    rhs_data: Tensor<T, RANK, Cpu>,
    /// Gradient of the left-hand side operand.
    lhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    /// Gradient of the right-hand side operand.
    rhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    /// Gradient of the output (received from the parent node).
    out_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    /// Parent nodes in the computation graph.
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
// --- Div Node ---
/// A node representing element-wise division in the computation graph.
#[derive(Debug)]
struct DivNode<T: TensorElem, const RANK: usize> {
    lhs_data: Tensor<T, RANK, Cpu>,
    rhs_data: Tensor<T, RANK, Cpu>,
    lhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    rhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    out_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    parents: Vec<Rc<dyn GraphNode>>,
}

impl<T: TensorElem, const RANK: usize> GraphNode for DivNode<T, RANK> {
    fn backward(&self) {
        if let Some(grad) = self.out_grad.borrow().as_ref() {
            // d(x/y)/dx = 1/y * grad
            // d(x/y)/dy = -x/y^2 * grad

            {
                let mut lhs = self.lhs_grad.borrow_mut();
                let dl_dx = (grad.div(&self.rhs_data)).unwrap();
                if let Some(l) = lhs.as_mut() {
                    *l = (l.add(&dl_dx)).unwrap();
                } else {
                    *lhs = Some(dl_dx);
                }
            }

            {
                let mut rhs = self.rhs_grad.borrow_mut();
                // -x / y^2 * grad
                // = -1 * x * y^-2 * grad
                // = -(x * grad) / (y * y)

                let x_grad = (&self.lhs_data * grad).unwrap();
                let y_sq = (&self.rhs_data * &self.rhs_data).unwrap();
                let val = (x_grad.div(&y_sq)).unwrap();

                let zero = Tensor::zeros(*val.shape());
                let neg_val = (zero.sub(&val)).unwrap();

                if let Some(r) = rhs.as_mut() {
                    *r = (r.add(&neg_val)).unwrap();
                } else {
                    *rhs = Some(neg_val);
                }
            }
        }
    }

    fn parents(&self) -> Vec<Rc<dyn GraphNode>> {
        self.parents.clone()
    }
}

impl<T: TensorElem + 'static, const RANK: usize> Div for Variable<T, RANK> {
    type Output = Variable<T, RANK>;

    fn div(self, rhs: Self) -> Self::Output {
        let data = (&self.data / &rhs.data).unwrap();

        let mut parents = Vec::new();
        if let Some(p) = &self.node {
            parents.push(p.clone());
        }
        if let Some(p) = &rhs.node {
            parents.push(p.clone());
        }

        let out_grad = Rc::new(RefCell::new(None));

        let node = Rc::new(DivNode {
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
    /// Data of the left-hand side operand.
    lhs_data: Tensor<T, RANK, Cpu>,
    /// Data of the right-hand side operand.
    rhs_data: Tensor<T, RANK, Cpu>,
    /// Gradient of the left-hand side operand.
    lhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    /// Gradient of the right-hand side operand.
    rhs_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    /// Gradient of the output.
    out_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    /// Parent nodes in the computation graph.
    parents: Vec<Rc<dyn GraphNode>>,
}

impl<T: TensorElem, const RANK: usize> GraphNode for MatMulNode<T, RANK> {
    #[allow(clippy::collapsible_if)]
    fn backward(&self) {
        if let Some(grad) = self.out_grad.borrow().as_ref() {
            // LHS Gradient
            {
                let mut lhs = self.lhs_grad.borrow_mut();
                let rhs_t = self.rhs_data.transpose().unwrap();
                let dl_da = grad.matmul(&rhs_t).unwrap();
                if let Some(l) = lhs.as_mut() {
                    *l = (l.add(&dl_da)).unwrap();
                } else {
                    *lhs = Some(dl_da);
                }
            }

            // RHS Gradient
            {
                let mut rhs = self.rhs_grad.borrow_mut();
                let lhs_t = self.lhs_data.transpose().unwrap();
                let dr_db = lhs_t.matmul(grad).unwrap();
                if let Some(r) = rhs.as_mut() {
                    *r = (r.add(&dr_db)).unwrap();
                } else {
                    *rhs = Some(dr_db);
                }
            }
        }
    }

    fn parents(&self) -> Vec<Rc<dyn GraphNode>> {
        self.parents.clone()
    }
}

// --- Exp Node ---
#[derive(Debug)]
struct ExpNode<T: TensorElem, const RANK: usize> {
    out_data: Tensor<T, RANK, Cpu>,
    input_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    out_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    parents: Vec<Rc<dyn GraphNode>>,
}

impl<T: TensorElem + 'static, const RANK: usize> GraphNode for ExpNode<T, RANK> {
    fn backward(&self) {
        if let Some(grad) = self.out_grad.borrow().as_ref() {
            // d(exp(x))/dx = exp(x) * grad = out * grad
            //
            // Explanation:
            // The derivative of e^x is e^x.
            // By chain rule, if y = e^x, then dL/dx = dL/dy * dy/dx = grad * e^x.
            // Since out_data = e^x, we can reuse it: dL/dx = grad * out_data.
            let mut input = self.input_grad.borrow_mut();
            let dx = (&self.out_data * grad).unwrap();
            if let Some(i) = input.as_mut() {
                *i = (i.add(&dx)).unwrap();
            } else {
                *input = Some(dx);
            }
        }
    }

    fn parents(&self) -> Vec<Rc<dyn GraphNode>> {
        self.parents.clone()
    }
}

// --- Log Node ---
#[derive(Debug)]
struct LogNode<T: TensorElem, const RANK: usize> {
    input_data: Tensor<T, RANK, Cpu>,
    input_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    out_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    parents: Vec<Rc<dyn GraphNode>>,
}

impl<T: TensorElem + 'static, const RANK: usize> GraphNode for LogNode<T, RANK> {
    fn backward(&self) {
        if let Some(grad) = self.out_grad.borrow().as_ref() {
            // d(log(x))/dx = 1/x * grad
            //
            // Explanation:
            // The derivative of ln(x) is 1/x.
            // By chain rule: dL/dx = dL/dy * dy/dx = grad * (1/x) = grad / x.
            let mut input = self.input_grad.borrow_mut();
            // 1/x * grad = grad / x
            let dx = (grad.div(&self.input_data)).unwrap();
            if let Some(i) = input.as_mut() {
                *i = (i.add(&dx)).unwrap();
            } else {
                *input = Some(dx);
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

    pub fn exp(&self) -> Self {
        // Assuming T has exp (Float). TensorElem includes Num, but maybe not Float directly in trait bounds?
        // We might need to use map with f64 cast if T doesn't support exp directly.
        // Or assume T is f32/f64.
        // Let's use map.
        let data = self
            .data
            .map(|x| T::from_f64(x.to_f64().unwrap().exp()).unwrap());

        let mut parents = Vec::new();
        if let Some(p) = &self.node {
            parents.push(p.clone());
        }

        let out_grad = Rc::new(RefCell::new(None));

        let node = Rc::new(ExpNode {
            out_data: data.clone(),
            input_grad: self.grad.clone(),
            out_grad: out_grad.clone(),
            parents,
        });

        Variable {
            data,
            grad: out_grad,
            node: Some(node),
        }
    }

    pub fn log(&self) -> Self {
        let data = self
            .data
            .map(|x| T::from_f64(x.to_f64().unwrap().ln()).unwrap());

        let mut parents = Vec::new();
        if let Some(p) = &self.node {
            parents.push(p.clone());
        }

        let out_grad = Rc::new(RefCell::new(None));

        let node = Rc::new(LogNode {
            input_data: self.data.clone(),
            input_grad: self.grad.clone(),
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

    #[test]
    fn test_matmul_backward() {
        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 6], [7, 8]] (2x2)
        // C = A @ B
        // C = [[19, 22], [43, 50]]

        // Let Loss L = sum(C) = 19 + 22 + 43 + 50 = 134
        // dL/dC = [[1, 1], [1, 1]]

        // dL/dA = dL/dC @ B^T
        //       = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]]
        //       = [[11, 15], [11, 15]]

        // dL/dB = A^T @ dL/dC
        //       = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]]
        //       = [[4, 4], [6, 6]]

        let a_data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let b_data = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], [2, 2]).unwrap();

        let a = Variable::new(a_data);
        let b = Variable::new(b_data);

        let c = a.matmul(&b).unwrap();

        // Manually seed gradient with ones (equivalent to sum(C))
        *c.grad.borrow_mut() = Some(Tensor::ones([2, 2]));

        // We need to manually trigger backward on the node because c is not a scalar
        // and Variable::backward() assumes scalar and seeds with 1.0.
        // But here we want to test the MatMulNode backward specifically.
        // However, Variable::backward() calls engine::backward(self.node).
        // If we seed grad manually, we can call c.backward() but we need to be careful
        // that it doesn't overwrite our seed.
        // Variable::backward() checks `if self.grad.borrow().is_none()`.
        // So if we seed it first, it should be fine.

        c.backward();

        let a_grad = a.grad.borrow().as_ref().unwrap().clone();
        let b_grad = b.grad.borrow().as_ref().unwrap().clone();

        assert_eq!(a_grad.data(), &[11.0, 15.0, 11.0, 15.0]);
        assert_eq!(b_grad.data(), &[4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn test_matmul_chain_rule() {
        // y = sum( (A @ x) * x )
        // A = [[1, 2], [3, 4]]
        // x = [1, 2]
        // A@x = [5, 11]
        // (A@x)*x = [5, 22]
        // y = 27

        // This is a bit complex to derive manually quickly.
        // Let's try a simpler one: y = sum(A @ x)
        // A = [[1, 2], [3, 4]]
        // x = [1, 2]
        // A@x = [5, 11]
        // y = 16

        // dy/dx = A^T @ ones
        //       = [[1, 3], [2, 4]] @ [1, 1]
        //       = [4, 6]

        // dy/dA = ones @ x^T (outer product)
        //       = [1, 1] @ [1, 2]
        //       = [[1, 2], [1, 2]]

        let a_data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let x_data = Tensor::new(vec![1.0, 2.0], [2, 1]).unwrap(); // Column vector

        let a = Variable::new(a_data);
        let x = Variable::new(x_data);

        let y_vec = a.matmul(&x).unwrap();

        // To make it a scalar for easy backward:
        // We don't have a "sum" operation in autograd yet.
        // But we can simulate "sum" by doing dot product with ones, or just seeding gradient with ones.
        // Let's seed gradient of y_vec with ones.

        *y_vec.grad.borrow_mut() = Some(Tensor::ones([2, 1]));
        y_vec.backward();

        let x_grad = x.grad.borrow().as_ref().unwrap().clone();
        let a_grad = a.grad.borrow().as_ref().unwrap().clone();

        assert_eq!(x_grad.data(), &[4.0, 6.0]);
        assert_eq!(a_grad.data(), &[1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_add_accumulation() {
        // y = x + x + x
        // x = 3
        // y = 9
        // dy/dx = 3

        let x = Variable::new(Tensor::new(vec![3.0], []).unwrap());
        let y = x.clone() + x.clone() + x.clone();

        y.backward();

        assert_eq!(x.grad.borrow().as_ref().unwrap().data()[0], 3.0);
    }

    #[test]
    fn test_mul_accumulation() {
        // y = x * x * x
        // x = 3
        // y = 27
        // dy/dx = 3x^2 = 27

        let x = Variable::new(Tensor::new(vec![3.0], []).unwrap());
        let y = x.clone() * x.clone() * x.clone();

        y.backward();

        assert_eq!(x.grad.borrow().as_ref().unwrap().data()[0], 27.0);
    }

    #[test]
    fn test_matmul_accumulation() {
        // Y = X @ X @ X
        // X = [[1, 0], [0, 1]] (Identity)
        // Y = I
        // Loss = sum(Y) = 2
        // dL/dX should be 3 * I ?
        // Let's use scalar logic for intuition: y = x^3, dy/dx = 3x^2. If x=1, dy/dx=3.
        // For matrix: d(X^3)/dX.
        // If X = I, X^2 = I, X^3 = I.
        // dL/dX = 3 * X^2 = 3 * I.

        let x_data = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], [2, 2]).unwrap();
        let x = Variable::new(x_data);

        let y = x.matmul(&x).unwrap().matmul(&x).unwrap();

        *y.grad.borrow_mut() = Some(Tensor::ones([2, 2]));
        y.backward();

        let x_grad = x.grad.borrow().as_ref().unwrap().clone();
        // Expected gradient is 3 * ones (since we seeded with ones and dY/dX is 3*I effectively distributed)
        // Wait.
        // Y = X^3. L = sum(Y).
        // dL/dX = 3 * (X^T)^2 @ Ones?
        // Let's just check the result.
        // dL/dX = [[3, 3], [3, 3]]

        assert_eq!(x_grad.data(), &[3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_non_leaf_operations() {
        // Test operations where RHS has a node (is not a leaf)
        // This ensures coverage for `if let Some(p) = &rhs.node` branches

        let x = Variable::new(Tensor::new(vec![2.0], []).unwrap());

        // a has a node
        let a = x.clone() * x.clone(); // 4.0

        // b = a + a. RHS a has node.
        let b = a.clone() + a.clone(); // 8.0

        // c = a * a. RHS a has node.
        let c = a.clone() * a.clone(); // 16.0

        // d = a @ a. RHS a has node. (Need rank 2 for matmul)
        let m = Variable::new(Tensor::new(vec![2.0], [1, 1]).unwrap());
        let n = m.matmul(&m).unwrap(); // n has node
        let _p = n.matmul(&n).unwrap(); // RHS n has node

        b.backward();
        c.backward();
        // We don't check gradients here, just ensuring the code paths run.
    }
    #[test]
    fn test_sub_backward() {
        let a = Variable::new(Tensor::new(vec![5.0], []).unwrap());
        let b = Variable::new(Tensor::new(vec![3.0], []).unwrap());
        let c = a.clone() - b.clone();

        c.backward();

        assert_eq!(a.grad.borrow().as_ref().unwrap().data()[0], 1.0);
        assert_eq!(b.grad.borrow().as_ref().unwrap().data()[0], -1.0);
    }

    #[test]
    fn test_div_backward() {
        // y = a / b
        // a = 6, b = 3
        // y = 2
        // dy/da = 1/b = 1/3
        // dy/db = -a/b^2 = -6/9 = -2/3

        let a = Variable::new(Tensor::new(vec![6.0], []).unwrap());
        let b = Variable::new(Tensor::new(vec![3.0], []).unwrap());
        let c = a.clone() / b.clone();

        c.backward();

        let a_grad = a.grad.borrow().as_ref().unwrap().data()[0];
        let b_grad = b.grad.borrow().as_ref().unwrap().data()[0];

        assert!((a_grad - (1.0f64 / 3.0)).abs() < 1e-6);
        assert!((b_grad - (-2.0f64 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_exp_backward() {
        // y = exp(x)
        // x = 0
        // y = 1
        // dy/dx = exp(x) = 1

        let x = Variable::new(Tensor::new(vec![0.0], []).unwrap());
        let y = x.exp();

        y.backward();

        assert_eq!(x.grad.borrow().as_ref().unwrap().data()[0], 1.0);
    }

    #[test]
    fn test_log_backward() {
        // y = log(x)
        // x = 2
        // dy/dx = 1/x = 0.5

        let x = Variable::new(Tensor::new(vec![2.0], []).unwrap());
        let y = x.log();

        y.backward();

        assert_eq!(x.grad.borrow().as_ref().unwrap().data()[0], 0.5);
    }

    #[test]
    fn test_sub_accumulation() {
        // y = x - x
        // dy/dx = 1 - 1 = 0

        let x = Variable::new(Tensor::new(vec![3.0], []).unwrap());
        let y = x.clone() - x.clone();

        y.backward();

        assert_eq!(x.grad.borrow().as_ref().unwrap().data()[0], 0.0);
    }

    #[test]
    fn test_sub_accumulation_complex() {
        // y = (x - a) - x
        // dy/dx = 1 - 1 = 0
        let x = Variable::new(Tensor::new(vec![3.0], []).unwrap());
        let a = Variable::new(Tensor::new(vec![1.0], []).unwrap());
        let y = (x.clone() - a) - x.clone();

        y.backward();
        assert_eq!(x.grad.borrow().as_ref().unwrap().data()[0], 0.0);
    }

    #[test]
    fn test_div_accumulation() {
        // y = x / x = 1
        // dy/dx = 0

        let x = Variable::new(Tensor::new(vec![3.0], []).unwrap());
        let y = x.clone() / x.clone();

        y.backward();

        assert_eq!(x.grad.borrow().as_ref().unwrap().data()[0], 0.0);
    }

    #[test]
    fn test_div_accumulation_complex() {
        // y = (x / a) / x = 1/a
        // dy/dx = 0
        // Wait. y = (x/a) * x^-1
        // dy/dx = (1/a)*x^-1 + (x/a)*(-x^-2)
        //       = 1/(ax) - 1/(ax) = 0.
        let x = Variable::new(Tensor::new(vec![3.0], []).unwrap());
        let a = Variable::new(Tensor::new(vec![1.0], []).unwrap());
        let y = (x.clone() / a) / x.clone();

        y.backward();
        assert_eq!(x.grad.borrow().as_ref().unwrap().data()[0], 0.0);
    }

    #[test]
    fn test_exp_accumulation() {
        // y = exp(x) + exp(x) = 2exp(x)
        // x = 0
        // dy/dx = 2exp(0) = 2

        let x = Variable::new(Tensor::new(vec![0.0], []).unwrap());
        let y = x.exp() + x.exp();

        y.backward();

        assert_eq!(x.grad.borrow().as_ref().unwrap().data()[0], 2.0);
    }

    #[test]
    fn test_log_accumulation() {
        // y = log(x) + log(x) = 2log(x)
        // x = 2
        // dy/dx = 2/x = 1

        let x = Variable::new(Tensor::new(vec![2.0], []).unwrap());
        let y = x.log() + x.log();

        y.backward();

        assert_eq!(x.grad.borrow().as_ref().unwrap().data()[0], 1.0);
    }
}
