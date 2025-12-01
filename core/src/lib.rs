//! # xla-rs
//!
//! `xla-rs` is a pure Rust implementation of tensor operations and neural network building blocks,
//! designed for **educational purposes** to help students and engineers understand the internals of LLM inference.
//!
//! Unlike production frameworks (PyTorch, TensorFlow) that wrap C++/CUDA kernels, `xla-rs` implements
//! everything in readable Rust. This makes it an excellent resource for learning how deep learning
//! actually works "under the hood."
//!
//! > [!NOTE]
//! > **Educational vs. Production**: This crate prioritizes code readability and simplicity over maximum performance.
//! > While we use some optimizations (like `rayon` for parallelism), it is not intended to replace
//! > optimized frameworks for large-scale training.
//!
//! ## Modules
//!
//! - [`mod@tensor`]: The foundation. N-dimensional arrays with broadcasting and arithmetic.
//! - [`autograd`]: Automatic differentiation engine (reverse-mode) for training.
//! - [`nn`]: Neural network layers (Linear, RMSNorm, MoE, etc.).
//! - `models`: Full model architectures (e.g., Gemma, GPT-2) (requires `models` feature).
//!
//! ## Quick Start
//!
//! Here's how to create a tensor and perform a simple operation:
//!
//! ```rust
//! use xla_rs::tensor::Tensor;
//!
//! // 1. Create a 2D tensor (matrix)
//! let data = vec![1.0, 2.0, 3.0, 4.0];
//! let tensor = Tensor::<f32, 2>::new(data, [2, 2]).unwrap();
//! println!("Tensor:\n{:?}", tensor);
//!
//! // 2. Zero-overhead compile-time operations (Expert Feature)
//! // `xla-rs` allows defining tensors and operations that are evaluated at compile-time!
//! use xla_rs::tensor::ConstDevice;
//!
//! const A: Tensor<f32, 2, ConstDevice<4>> = Tensor::new_const([1.0, 2.0, 3.0, 4.0], [2, 2]);
//! const B: Tensor<f32, 2, ConstDevice<4>> = A.transpose(); // Evaluated at compile time!
//! ```
//!
//! > [!TIP]
//! > **Expert Note on `ConstDevice`**: The `ConstDevice` allows us to burn-in weights or pre-compute
//! > constants (like positional embeddings) directly into the binary, potentially reducing startup time
//! > and runtime overhead.

/// Macro for creating a Tensor with compile-time shape checking.
///
/// # Examples
///
/// ```rust
/// use xla_rs::tensor;
/// use xla_rs::tensor::Tensor;
///
/// // Works
/// let t = tensor!([1.0, 2.0, 3.0, 4.0], [2, 2]);
///
/// // Fails to compile:
/// // let t = tensor!([1.0, 2.0, 3.0], [2, 2]);
/// ```
#[macro_export]
macro_rules! tensor {
    ($data:expr, $shape:expr) => {{
        // Constants to force compile-time evaluation
        const DATA_LEN: usize = (&$data as &[_]).len();
        const SHAPE: [usize; (&$shape as &[_]).len()] = $shape;
        const EXPECTED_SIZE: usize = {
            let mut size = 1;
            let mut i = 0;
            while i < (&SHAPE as &[_]).len() {
                size *= SHAPE[i];
                i += 1;
            }
            size
        };

        // This assertion triggers a compile-time error if false
        const _: () = assert!(
            DATA_LEN == EXPECTED_SIZE,
            "Shape mismatch: data length does not match shape product"
        );

        // Safe to unwrap because we checked at compile time
        $crate::tensor::Tensor::new($data.to_vec(), $shape).unwrap()
    }};
}

pub mod autograd;
pub mod checkpoint;
pub mod data;
pub mod distributed;
pub use autograd::Variable;
pub use tensor::{ConstDevice, Cpu, Device, Storage, Tensor, TensorElem, TensorError, TensorOps};

pub mod kernels;
pub mod loss;
#[cfg(feature = "models")]
pub mod models;
pub mod nn;
pub mod optim;
pub mod tensor;
