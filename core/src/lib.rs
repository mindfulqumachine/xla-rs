//! # xla-rs
//!
//! `xla-rs` is a pure Rust implementation of tensor operations and neural network building blocks,
//! designed for educational purposes and understanding the internals of LLM inference.
//!
//! Despite the name, it currently runs on **CPU only** and does not yet integrate with the XLA compiler.
//!
//! ## Modules
//!
//! - [`mod@tensor`]: Core N-dimensional tensor implementation.
//! - [`nn`]: Neural network layers (Linear, RMSNorm, MoE, etc.).
//! - `models`: Model architectures (e.g., Gemma) (requires `models` feature).
//!
//! ## Example
//!
//! ```rust
//! use xla_rs::tensor::Tensor;
//!
//! let data = vec![1.0, 2.0, 3.0, 4.0];
//! let tensor = Tensor::<f32, 2>::new(data, [2, 2]).unwrap();
//! println!("{:?}", tensor);
//!
//! // Zero-overhead compile-time operations
//! use xla_rs::tensor::ConstDevice;
//!
//! const A: Tensor<f32, 2, ConstDevice<4>> = Tensor::new_const([1.0, 2.0, 3.0, 4.0], [2, 2]);
//! const B: Tensor<f32, 2, ConstDevice<4>> = A.transpose(); // Evaluated at compile time!
//! ```

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
