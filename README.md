# xla-rs

**xla-rs** is a pure Rust implementation of tensor operations and neural network building blocks, designed for educational purposes.

> [!NOTE]
> Despite the name, this project currently runs on **CPU only** and does not integrate with the XLA compiler. It serves as a playground for understanding the internals of LLM inference (specifically Gemma) in Rust.

## Features

- **Pure Rust**: No C++ dependencies, built on top of `Vec<T>` and `rayon` for parallelism.
- **Tensor Library**: N-dimensional tensors with broadcasting, reshaping, and matrix multiplication.
- **Neural Networks**: Implementation of `Linear`, `RMSNorm`, `SiLU`, and `MoE` layers.
- **Gemma Architecture**: Full implementation of the Gemma transformer model.

## Usage

### Creating Tensors

```rust
use xla_rs::tensor::Tensor;

fn main() {
    // Create a 2x2 tensor
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<f32, 2>::new(data, [2, 2]).unwrap();
    
    println!("Tensor: {:?}", tensor);
}
```

### Running a Linear Layer

```rust
use xla_rs::tensor::Tensor;
use xla_rs::nn::{Linear, Module};

fn main() {
    // Input: [Batch=1, InputDim=2]
    let input = Tensor::<f32, 2>::new(vec![1.0, 2.0], [1, 2]).unwrap();
    
    // Weights: [OutputDim=2, InputDim=2]
    let weights = Tensor::<f32, 2>::new(vec![0.5, 0.5, 0.5, 0.5], [2, 2]).unwrap();
    
    let linear = Linear::new(weights, None);
    let output = linear.forward(&input).unwrap();
    
    println!("Output: {:?}", output);
}
```

## Project Structure

- `src/tensor`: Core tensor implementation (`Tensor`, `Device`, `Storage`).
- `src/nn`: Neural network modules (`Linear`, `RMSNorm`, `MoE`).
- `src/models`: Model architectures (`Gemma`).

## Future Roadmap

- [ ] Autograd support (Backpropagation)
- [ ] Quantization (int8, fp8)
- [ ] XLA / GPU integration
