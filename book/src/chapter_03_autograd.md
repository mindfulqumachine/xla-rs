# Chapter 3: Automatic Differentiation

Training neural networks requires calculating gradients of a loss function with respect to the model's parameters. We use **Automatic Differentiation (Autograd)** to do this efficiently and automatically.

## What is Autograd?

Automatic Differentiation (Autograd) is a set of techniques to numerically evaluate the derivative of a function specified by a computer program. Unlike numerical differentiation (which approximates derivatives) or symbolic differentiation (which manipulates mathematical expressions), autograd exploits the fact that every computer program, no matter how complicated, executes a sequence of elementary arithmetic operations (addition, subtraction, multiplication, division, etc.) and elementary functions (exp, log, sin, cos, etc.). By applying the chain rule repeatedly to these operations, derivatives of arbitrary order can be computed automatically, accurately to working precision, and using at most a small constant factor more arithmetic operations than the original program.

## Why is it Important?

In Deep Learning, we train models by minimizing a loss function. Optimization algorithms like **Gradient Descent** require the gradient of the loss with respect to the weights. For modern deep neural networks with millions or billions of parameters, manually deriving and implementing these gradients is error-prone and impractical. Autograd allows us to:
1.  **Iterate quickly**: We can change the model architecture without rewriting the backward pass.
2.  **Correctness**: It guarantees accurate gradients (within floating-point precision).
3.  **Efficiency**: It computes gradients efficiently, often sharing intermediate computations.

### Practical Example: Linear Regression Step

Consider a simple linear regression model \\(y = w \cdot x + b\\). We want to find \\(w\\) and \\(b\\) that minimize the squared error loss \\(L = (y - \text{target})^2\\).

Without autograd, you would need to manually derive:
$$ \frac{\partial L}{\partial w} = 2(wx + b - \text{target}) \cdot x $$
$$ \frac{\partial L}{\partial b} = 2(wx + b - \text{target}) $$

With autograd, you simply define the forward pass, and the system handles the rest:

```rust
# extern crate xla_rs;
use xla_rs::tensor::Tensor;
use xla_rs::autograd::Variable;

// Inputs
let x = Variable::new(Tensor::new(vec![2.0], []).unwrap());
let target = Variable::new(Tensor::new(vec![10.0], []).unwrap());

// Parameters (initialized randomly, here fixed for example)
let w = Variable::new(Tensor::new(vec![3.0], []).unwrap());
let b = Variable::new(Tensor::new(vec![1.0], []).unwrap());

// Forward pass: y = w * x + b
let y = w.clone() * x.clone() + b.clone();

// Loss: (y - target)^2
// Note: We don't have Pow yet, so we multiply by itself
let diff = y + Variable::new(Tensor::new(vec![-1.0], []).unwrap()) * target; // y - target
let loss = diff.clone() * diff.clone(); // (y - target)^2

// Backward pass
loss.backward();

// Gradients are automatically populated!
// y = 3*2 + 1 = 7
// diff = 7 - 10 = -3
// loss = (-3)^2 = 9
// dL/dw = 2 * diff * x = 2 * -3 * 2 = -12
assert_eq!(w.grad.borrow().as_ref().unwrap().data()[0], -12.0);
```

## Implementation in xla-rs

We implement a **Define-by-Run** (or Tape-based) autograd system, similar to PyTorch. This means the computation graph is built dynamically as you perform operations.

The core logic is located in:
-   [`src/autograd/mod.rs`](file:///Users/blitz/my-oss/xla-rs/src/autograd/mod.rs): Defines the `Variable` struct and `GraphNode` trait.
-   [`src/autograd/engine.rs`](file:///Users/blitz/my-oss/xla-rs/src/autograd/engine.rs): Implements the backward pass engine (topological sort and graph traversal).
-   [`src/autograd/ops.rs`](file:///Users/blitz/my-oss/xla-rs/src/autograd/ops.rs): Implements the `GraphNode` trait for specific operations.
-   [`src/autograd/functional.rs`](file:///Users/blitz/my-oss/xla-rs/src/autograd/functional.rs): Implements functional API transformations like `grad`.

## The Tape

The "Tape" is a metaphor for recording the sequence of operations. In our implementation:

1.  **Variable**: This is the primary user-facing struct. It wraps a `Tensor` and tracks:
    -   `data`: The actual tensor data.
    -   `grad`: The gradient of the loss with respect to this variable.
    -   `node`: A reference to the `GraphNode` that produced this variable.

2.  **GraphNode**: A trait representing an operation in the computation graph. It stores:
    -   References to input variables' gradients.
    -   References to parent nodes.
    -   The `backward()` method.

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

When you call `.backward()` on a scalar variable (typically the loss), the following happens:

1.  **Seed Gradient**: We initialize the gradient of the loss variable to 1.0.
2.  **Topological Sort**: We traverse the graph backwards to find all dependencies and sort them topologically.
3.  **Backpropagation**: We iterate through the sorted nodes and call `.backward()` on each.

## Functional API (JAX-style)

While the object-oriented `.backward()` style (like PyTorch) is intuitive, functional transformations (like JAX) offer a powerful alternative. We provide a `grad` function that takes a function and returns a new function that computes the gradient.

This approach is "purely functional" in the sense that it doesn't rely on side effects (mutating `.grad` fields) visible to the user, but rather returns the gradients directly.

```rust
# extern crate xla_rs;
use xla_rs::tensor::Tensor;
use xla_rs::autograd::{Variable, functional};

fn square(x: Variable<f32, 0>) -> Variable<f32, 0> {
    x.clone() * x.clone()
}

let x_data = Tensor::new(vec![3.0], []).unwrap();

// Get the gradient function
let grad_square = functional::grad(square);

// Compute gradient at x = 3.0
// d(x^2)/dx = 2x = 6.0
let dx = grad_square(x_data);

assert_eq!(dx.data()[0], 6.0);
```

We also provide `value_and_grad` to get both the output and the gradient in one pass:

```rust
# extern crate xla_rs;
# use xla_rs::tensor::Tensor;
# use xla_rs::autograd::{Variable, functional};
# fn square(x: Variable<f32, 0>) -> Variable<f32, 0> {
#     x.clone() * x.clone()
# }
# let x_data = Tensor::new(vec![3.0], []).unwrap();
let (val, dx) = functional::value_and_grad(square)(x_data);

assert_eq!(val.data()[0], 9.0);
assert_eq!(dx.data()[0], 6.0);
```
