//! Core Tensor implementation.
//!
//! This module defines the `Tensor` struct, which is the central data structure in `xla-rs`.
//! It supports N-dimensional arrays, automatic differentiation (via the `autograd` module),
//! and various mathematical operations.
//!
//! # Key Components
//!
//! - [`Tensor`]: The main struct representing an N-dimensional array.
//! - [`TensorError`]: Error type for tensor operations.
//! - [`TensorElem`]: Trait bound for elements that can be stored in a tensor.
//!
//! # Examples
//!
//! ```rust
//! use xla_rs::tensor::Tensor;
//!
//! let data = vec![1.0, 2.0, 3.0, 4.0];
//! let tensor = Tensor::<f32, 2>::new(data, [2, 2]).unwrap();
//! assert_eq!(tensor.shape(), &[2, 2]);
//! ```

use num_traits::{FromPrimitive, Num, NumAssign, ToPrimitive};
use std::fmt::Debug;
use thiserror::Error;

pub mod device;
pub mod ops;
pub mod storage;

pub use device::{Cpu, Device};
pub use storage::Storage;

/// Error type for Tensor operations.
#[derive(Error, Debug)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("Incompatible shapes for broadcasting: {0:?} and {1:?}")]
    BroadcastError(Vec<usize>, Vec<usize>),
    #[error("Index out of bounds: index {index:?} for shape {shape:?}")]
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    #[error("Device mismatch")]
    DeviceMismatch,
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, TensorError>;

/// Trait bound for elements that can be stored in a Tensor.
///
/// # Requirements
/// - `Copy + Clone`: Essential for efficient storage in contiguous memory (e.g., `Vec<T>`) and fast element access.
/// - `Num + ...`: Provides necessary numeric operations for tensor math.
/// - `Send + Sync`: Required for parallel execution via `rayon`.
pub trait TensorElem:
    Num + NumAssign + Copy + Clone + Debug + Send + Sync + FromPrimitive + ToPrimitive + PartialOrd
{
}

impl<T> TensorElem for T where
    T: Num
        + NumAssign
        + Copy
        + Clone
        + Debug
        + Send
        + Sync
        + FromPrimitive
        + ToPrimitive
        + PartialOrd
{
}

/// The core Tensor struct.
///
/// Represents an N-dimensional array of elements.
///
/// # Generics
///
/// - `T`: The element type (must implement `TensorElem`).
/// - `RANK`: The number of dimensions (const generic).
/// - `D`: The device where data is stored (defaults to `Cpu`).
#[derive(Clone)]
pub struct Tensor<T, const RANK: usize, D: Device = Cpu>
where
    T: TensorElem,
{
    shape: [usize; RANK],
    strides: [usize; RANK],
    data: D::Storage<T>,
    device: D,
}

impl<T, const RANK: usize> Tensor<T, RANK, Cpu>
where
    T: TensorElem,
{
    /// Creates a new Tensor from a vector of data and a shape.
    ///
    /// # Arguments
    ///
    /// * `data` - A flat vector containing the tensor elements.
    /// * `shape` - An array representing the dimensions of the tensor.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ShapeMismatch` if the length of `data` does not match the product of `shape`.
    pub fn new(data: Vec<T>, shape: [usize; RANK]) -> Result<Self> {
        let size: usize = shape.iter().product();
        if data.len() != size {
            return Err(TensorError::ShapeMismatch {
                expected: vec![size],
                got: vec![data.len()],
            });
        }

        let strides = compute_strides(&shape);
        Ok(Self {
            shape,
            strides,
            data,
            device: Cpu,
        })
    }

    /// Creates a new Tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array representing the dimensions of the tensor.
    pub fn zeros(shape: [usize; RANK]) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![T::zero(); size];
        let strides = compute_strides(&shape);
        Self {
            shape,
            strides,
            data,
            device: Cpu,
        }
    }

    /// Creates a new Tensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array representing the dimensions of the tensor.
    pub fn ones(shape: [usize; RANK]) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![T::one(); size];
        let strides = compute_strides(&shape);
        Self {
            shape,
            strides,
            data,
            device: Cpu,
        }
    }

    /// Reshapes the tensor to a new shape.
    ///
    /// The number of elements must remain the same.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The target shape.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ShapeMismatch` if the total number of elements in `new_shape`
    /// does not match the current size of the tensor.
    pub fn reshape<const NEW_RANK: usize>(
        self,
        new_shape: [usize; NEW_RANK],
    ) -> Result<Tensor<T, NEW_RANK, Cpu>> {
        let current_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();

        if current_size != new_size {
            return Err(TensorError::ShapeMismatch {
                expected: vec![current_size],
                got: vec![new_size],
            });
        }

        let strides = compute_strides(&new_shape);
        Ok(Tensor {
            shape: new_shape,
            strides,
            data: self.data,
            device: self.device,
        })
    }
}

/// Computes the strides for a given shape.
///
/// Strides represent the number of elements to skip in memory to move to the next element
/// along a specific dimension. This implementation assumes a row-major (C-style) memory layout.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor.
///
/// # Returns
///
/// An array of strides corresponding to the input shape.
fn compute_strides<const RANK: usize>(shape: &[usize; RANK]) -> [usize; RANK] {
    let mut strides = [0; RANK];
    let mut stride = 1;
    for i in (0..RANK).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

impl<T, const RANK: usize, D: Device> Tensor<T, RANK, D>
where
    T: TensorElem,
{
    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize; RANK] {
        &self.shape
    }

    /// Returns the strides of the tensor.
    pub fn strides(&self) -> &[usize; RANK] {
        &self.strides
    }

    /// Returns a reference to the underlying data as a slice.
    pub fn data(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Returns a mutable reference to the underlying data as a slice.
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Returns the total number of elements in the tensor.
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<T, const RANK: usize, D: Device> Debug for Tensor<T, RANK, D>
where
    T: TensorElem,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("device", &self.device.name())
            .field("data_len", &self.data.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        // Positive case
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::<f32, 2>::new(data.clone(), [2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), &data[..]);

        // Negative case: Size mismatch
        let err = Tensor::<f32, 2>::new(vec![1.0, 2.0, 3.0], [2, 2]);
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_zeros_ones() {
        let zeros = Tensor::<f32, 2>::zeros([2, 3]);
        assert_eq!(zeros.data(), &[0.0; 6]);

        let ones = Tensor::<f32, 2>::ones([2, 3]);
        assert_eq!(ones.data(), &[1.0; 6]);
    }

    #[test]
    fn test_reshape() {
        let tensor = Tensor::<f32, 2>::zeros([2, 3]); // 6 elements

        // Valid reshape
        let reshaped = tensor.reshape([3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);

        // Invalid reshape
        let err = reshaped.clone().reshape([4, 2]); // 8 elements
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_arithmetic() {
        let a = Tensor::<f32, 1>::new(vec![1.0, 2.0], [2]).unwrap();
        let b = Tensor::<f32, 1>::new(vec![3.0, 4.0], [2]).unwrap();

        // Add
        let c = (&a + &b).unwrap();
        assert_eq!(c.data(), &[4.0, 6.0]);

        // Mul
        let d = (&a * &b).unwrap();
        assert_eq!(d.data(), &[3.0, 8.0]);

        // Mismatch
        let _e = Tensor::<f32, 1>::new(vec![1.0], [1]).unwrap();
        let f = Tensor::<f32, 1>::new(vec![1.0, 2.0, 3.0], [3]).unwrap();
        let err = &a + &f;
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_matmul_2d() {
        // A: [2, 3], B: [3, 2] -> C: [2, 2]
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Tensor::<f32, 2>::new(a_data, [2, 3]).unwrap();

        let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0];
        let b = Tensor::<f32, 2>::new(b_data, [3, 2]).unwrap();

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);

        // Row 0: 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
        // Row 0, Col 1: 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
        // Row 1: 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
        // Row 1, Col 1: 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
        assert_eq!(c.data(), &[31.0, 19.0, 85.0, 55.0]);
    }

    #[test]
    fn test_matmul_broadcast_error() {
        let a = Tensor::<f32, 2>::zeros([2, 3]);
        let b = Tensor::<f32, 2>::zeros([4, 2]); // K mismatch (3 vs 4)

        let err = a.matmul(&b);
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_transpose() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::<f32, 2>::new(data, [2, 3]).unwrap();
        // [ 1 2 3 ]
        // [ 4 5 6 ]

        let t_t = t.transpose().unwrap();
        assert_eq!(t_t.shape(), &[3, 2]);
        // [ 1 4 ]
        // [ 2 5 ]
        // [ 3 6 ]
        assert_eq!(t_t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_axes() {
        // Rank 4 tensor [B, S, H, D] -> [B, H, S, D]
        // Shape: [1, 2, 2, 2] -> [1, 2, 2, 2] for simplicity but distinct values
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();

        let t = Tensor::<f32, 4>::new(data, [1, 2, 2, 2]).unwrap();

        let permuted = t.transpose_axes(1, 2).unwrap();
        assert_eq!(permuted.shape(), &[1, 2, 2, 2]); // H, S swapped but sizes same

        assert_eq!(permuted.data(), &[0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn test_macro() {
        let t = tensor!([1.0, 2.0, 3.0, 4.0], [2, 2]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
