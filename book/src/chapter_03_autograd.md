# Chapter 3: Automatic Differentiation

Training neural networks requires calculating gradients. We use **Automatic Differentiation (Autograd)** to do this efficiently.

## The Tape

We implement a **Define-by-Run** (or Tape-based) autograd system, similar to PyTorch.

1.  **Variable**: A wrapper around a `Tensor` that tracks its gradient and the operation that created it.
2.  **GraphNode**: A trait representing an operation in the computation graph.
3.  **Backward Pass**: We traverse the graph in reverse topological order to compute gradients.

```rust
# extern crate xla_rs;
# use std::rc::Rc;
# use std::cell::RefCell;
# use xla_rs::tensor::{Tensor, Cpu, TensorElem};
# use xla_rs::autograd::GraphNode;
pub struct Variable<T: TensorElem, const RANK: usize> {
    pub data: Tensor<T, RANK, Cpu>,
    pub grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    pub node: Option<Rc<dyn GraphNode>>,
}
```

## The Backward Pass

When you call `.backward()` on a scalar variable (the loss), we:
1.  Seed the gradient of the loss with 1.0.
2.  Topologically sort the graph (to ensure dependencies are processed first).
3.  Call `.backward()` on each node, propagating gradients to parents.

## Example

```rust
# extern crate xla_rs;
use xla_rs::tensor::Tensor;
use xla_rs::autograd::Variable;

let a = Variable::new(Tensor::new(vec![2.0], []).unwrap());
let b = Variable::new(Tensor::new(vec![3.0], []).unwrap());

// c = a * b
let c = a.clone() * b.clone(); 

c.backward();

// da = dc/da * grad_c = b * 1 = 3
// db = dc/db * grad_c = a * 1 = 2
```
