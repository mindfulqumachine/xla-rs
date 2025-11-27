# Chapter 2: Tensors (The Bedrock)

At the heart of every deep learning framework lies the **Tensor**. A tensor is simply a generalization of vectors and matrices to $N$ dimensions.

In `xla-rs`, our `Tensor` struct is defined as:

```rust
# extern crate xla_rs;
# use xla_rs::tensor::{Device, Cpu, TensorElem};
pub struct Tensor<T: TensorElem, const RANK: usize, D: Device = Cpu> {
    shape: [usize; RANK],
    strides: [usize; RANK],
    data: D::Storage<T>,
    device: D,
}
```

## Memory Layout

We store data in a single, contiguous `Vec<T>`. This is crucial for performance (cache locality) and interoperability with hardware accelerators.

### Shapes and Strides

How do we map an N-dimensional index $(i, j, k)$ to a flat index in the vector? We use **strides**.

The stride for a dimension tells us how many elements we need to skip in memory to move one step along that dimension.

For a tensor of shape `[2, 3]`:
- `strides[1]` (columns) is 1.
- `strides[0]` (rows) is 3 (the size of the next dimension).

The flat index is calculated as:
$$ \text{index} = \sum_{d=0}^{RANK-1} i_d \times \text{strides}[d] $$

## Operations

We implement basic arithmetic operations (`Add`, `Mul`) and Matrix Multiplication.

### Broadcasting

Broadcasting allows us to perform operations on tensors of different shapes.
> [!NOTE]
> Full broadcasting support (like NumPy) is currently a work in progress. For now, shapes must match exactly.

## Hands On

Let's create some tensors!

```rust
# extern crate xla_rs;
use xla_rs::tensor::Tensor;

let data = vec![1.0, 2.0, 3.0, 4.0];
let t = Tensor::<f32, 2>::new(data, [2, 2]).unwrap();
println!("{:?}", t);
```
