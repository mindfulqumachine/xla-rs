# Chapter 4: Neural Network Primitives

With tensors and autograd in place, we can build neural network layers.

## The Module Trait

We define a `Module` trait that all layers must implement. Currently, it's a marker trait for `Debug + Send + Sync`.

```rust
# extern crate xla_rs;
# use std::fmt::Debug;
# use xla_rs::tensor::TensorElem;
pub trait Module<T: TensorElem>: Debug + Send + Sync {}
```

## The Linear Layer

The `Linear` layer performs an affine transformation: $y = xA^T + b$.

It holds:
- `weight`: Shape `[out_features, in_features]`
- `bias`: Optional, Shape `[out_features]`

We implement `forward` to handle both 2D (matrix) and 3D (batched) inputs.

```rust
# extern crate xla_rs;
# use xla_rs::tensor::{Tensor, TensorElem, Cpu};
pub struct Linear<T: TensorElem> {
    pub weight: Tensor<T, 2, Cpu>,
    pub bias: Option<Tensor<T, 1, Cpu>>,
}
```

## Example

```rust
# extern crate xla_rs;
use xla_rs::tensor::Tensor;
use xla_rs::nn::Linear;

let weight = Tensor::<f32, 2>::ones([2, 2]);
let linear = Linear::new(weight, None);
let input = Tensor::<f32, 2>::ones([1, 2]);

let output = linear.forward(&input).unwrap();
```
