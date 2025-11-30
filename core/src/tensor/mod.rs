//! Core Tensor implementation.
//!
//! # What is a Tensor?
//!
//! In the context of deep learning, a **Tensor** is simply a multi-dimensional array. It is the
//! fundamental data structure used to store and manipulate data (inputs, weights, gradients).
//!
//! - **0D Tensor (Scalar)**: A single number (e.g., loss value).
//! - **1D Tensor (Vector)**: A list of numbers (e.g., a bias vector).
//! - **2D Tensor (Matrix)**: A grid of numbers (e.g., a weight matrix).
//! - **3D+ Tensor**: Higher-dimensional arrays (e.g., a batch of images: `[Batch, Height, Width, Channels]`).
//!
//! # How `xla-rs` Tensors Work
//!
//! In `xla-rs`, a `Tensor` is defined by:
//! 1. **Data**: A flat vector of elements (usually `f32`).
//! 2. **Shape**: An array of dimensions (e.g., `[2, 3]`).
//! 3. **Strides**: How to step through the flat data to traverse dimensions.
//!
//! ## Example: Creating and Inspecting a Tensor
//!
//! ```rust
//! use xla_rs::tensor::Tensor;
//!
//! // Create a 2x3 matrix (Rank 2 Tensor)
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let tensor = Tensor::<f32, 2>::new(data, [2, 3]).unwrap();
//!
//! assert_eq!(tensor.shape(), &[2, 3]);
//!
//! // Accessing data (flat slice)
//! assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! ```
//!
//! > [!TIP]
//! > **Expert Note: Strides and Memory Layout**
//! > `xla-rs` uses **Row-Major** (C-style) layout. This means the last dimension changes the fastest
//! > in memory. Understanding strides is crucial for implementing efficient operations like
//! > broadcasting and transposing without copying data.

use num_traits::{FromPrimitive, Num, NumAssign, ToPrimitive};
use std::fmt::Debug;
use thiserror::Error;

pub mod const_ops;
pub mod device;
pub mod ops;
pub mod storage;

pub use device::{ConstDevice, Cpu, Device};
pub use ops::TensorOps;
pub use storage::Storage;

/// Error type for Tensor operations.
#[derive(Error, Debug)]
pub enum TensorError {
    /// The shape of the data does not match the expected shape.
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// An index is out of bounds for the given shape.
    #[error("Index out of bounds: index {index:?} for shape {shape:?}")]
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    /// The requested operation is not supported (e.g., for a specific rank or type).
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
/// # Design Philosophy: `const RANK` vs `const SHAPE`
///
/// The `Tensor` struct uses `const RANK: usize` rather than encoding the full shape in the type system
/// (e.g., `Tensor<T, [32, 128]>`). This is a deliberate trade-off to balance **correctness** with **usability**.
///
/// **Why not `const SHAPE`?**
/// - **Dynamic Batching:** Deep learning models often need to handle variable batch sizes (e.g., training with batch size 32,
///   inference with batch size 1). Encoding shape in the type would require re-instantiating the model for every batch size.
/// - **Complexity:** Operations like `reshape` or `matmul` would require complex type-level arithmetic, making compiler
///   errors difficult to read and the API rigid.
///
/// **The Trade-off:**
/// - **Pros:** We gain the flexibility to handle variable sequence lengths and batch sizes without recompilation.
///   Function signatures remain readable (e.g., `fn forward(x: Tensor<f32, 2>)`).
/// - **Cons:** Shape mismatches (e.g., multiplying `[32, 10]` by `[5, 20]`) are caught at runtime rather than compile-time.
///
/// This approach aligns with the industry standard for most deep learning frameworks (like PyTorch, TensorFlow, and many Rust crates),
/// prioritizing the flexibility required for real-world model training and inference.
#[derive(Clone, Copy)]
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

impl<T, const RANK: usize, const N: usize> Tensor<T, RANK, ConstDevice<N>>
where
    T: TensorElem,
{
    /// Creates a new constant Tensor from an array.
    ///
    /// This function is `const`, allowing tensors to be defined at compile time.
    ///
    /// # Arguments
    ///
    /// * `data` - The array containing the tensor elements.
    /// * `shape` - The shape of the tensor.
    ///
    /// # Panics
    ///
    /// Panics at compile time if the number of elements in `data` (N) does not match the product of `shape`.
    pub const fn new_const(data: [T; N], shape: [usize; RANK]) -> Self {
        let mut size = 1;
        let mut i = 0;
        while i < RANK {
            size *= shape[i];
            i += 1;
        }

        assert!(
            N == size,
            "Shape mismatch: data length does not match shape product"
        );

        let strides = compute_strides(&shape);
        Self {
            shape,
            strides,
            data,
            device: ConstDevice,
        }
    }

    /// Reshapes the tensor to a new shape.
    ///
    /// This is a `const` operation for `ConstDevice`.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The target shape.
    ///
    /// # Panics
    ///
    /// Panics at compile time if the number of elements in `new_shape` does not match the current size.
    pub const fn reshape<const NEW_RANK: usize>(
        self,
        new_shape: [usize; NEW_RANK],
    ) -> Tensor<T, NEW_RANK, ConstDevice<N>> {
        let mut new_size = 1;
        let mut i = 0;
        while i < NEW_RANK {
            new_size *= new_shape[i];
            i += 1;
        }

        assert!(
            N == new_size,
            "Shape mismatch: new shape size does not match tensor size"
        );

        let strides = compute_strides(&new_shape);
        Tensor {
            shape: new_shape,
            strides,
            data: self.data,
            device: ConstDevice,
        }
    }

    /// Transposes the tensor.
    ///
    /// Swaps the last two dimensions.
    /// This is a `const` operation for `ConstDevice`.
    ///
    /// # Panics
    ///
    /// Panics at compile time if rank < 2.
    pub const fn transpose(self) -> Self {
        assert!(RANK >= 2, "Transpose requires rank >= 2");

        let mut new_shape = self.shape;
        // Swap last two dims
        let tmp = new_shape[RANK - 1];
        new_shape[RANK - 1] = new_shape[RANK - 2];
        new_shape[RANK - 2] = tmp;

        let new_strides = compute_strides(&new_shape);
        let mut new_data = [self.data[0]; N]; // Initialize with dummy value, will be overwritten

        // We need to iterate over all elements, compute their multi-dim index in original shape,
        // swap the last two indices, and compute the linear index in the new shape.
        let mut i = 0;
        while i < N {
            // 1. Linear index `i` -> Multi-dim index `coords`
            let mut coords = [0; RANK];
            let mut rem = i;
            let mut d = 0;
            while d < RANK {
                coords[d] = rem / self.strides[d];
                rem %= self.strides[d];
                d += 1;
            }

            // 2. Swap last two coords
            let tmp_coord = coords[RANK - 1];
            coords[RANK - 1] = coords[RANK - 2];
            coords[RANK - 2] = tmp_coord;

            // 3. Multi-dim index `coords` -> Linear index `j` (using new_strides)
            let mut j = 0;
            let mut k = 0;
            while k < RANK {
                j += coords[k] * new_strides[k];
                k += 1;
            }

            // 4. Copy data
            new_data[j] = self.data[i];
            i += 1;
        }

        Tensor {
            shape: new_shape,
            strides: new_strides,
            data: new_data,
            device: ConstDevice,
        }
    }
}

impl<const N: usize> Tensor<f32, 2, ConstDevice<N>> {
    /// Performs matrix multiplication at compile time.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side tensor.
    ///
    /// # Returns
    ///
    /// A new tensor resulting from the matrix multiplication.
    ///
    /// # Type Parameters
    ///
    /// * `N2`: The size (number of elements) of the `rhs` tensor.
    /// * `OUT_N`: The size (number of elements) of the output tensor.
    ///
    /// # Panics
    ///
    /// Panics at compile time (or runtime if called there) if shapes are incompatible
    /// or if the provided `OUT_N` does not match the expected output size.
    pub const fn matmul<const N2: usize, const OUT_N: usize>(
        &self,
        rhs: &Tensor<f32, 2, ConstDevice<N2>>,
    ) -> Tensor<f32, 2, ConstDevice<OUT_N>> {
        let dim0 = self.shape[0];
        let dim1 = self.shape[1];
        let rhs_dim0 = rhs.shape[0];
        let rhs_dim1 = rhs.shape[1];

        if dim1 != rhs_dim0 {
            panic!("Shape mismatch for matmul");
        }
        if dim0 * rhs_dim1 != OUT_N {
            panic!("Output size mismatch");
        }

        let mut new_data = [0.0; OUT_N];
        let mut i = 0;
        while i < dim0 {
            let mut j = 0;
            while j < rhs_dim1 {
                let mut sum = 0.0;
                let mut k = 0;
                while k < dim1 {
                    // self[i, k]
                    let idx_a = i * self.strides[0] + k * self.strides[1];
                    // rhs[k, j]
                    let idx_b = k * rhs.strides[0] + j * rhs.strides[1];

                    let val_a = self.data[idx_a];
                    let val_b = rhs.data[idx_b];

                    // sum += val_a * val_b
                    sum = crate::tensor::const_ops::const_f32_add(
                        sum,
                        crate::tensor::const_ops::const_f32_mul(val_a, val_b),
                    );

                    k += 1;
                }
                new_data[i * rhs_dim1 + j] = sum;
                j += 1;
            }
            i += 1;
        }

        let new_shape = [dim0, rhs_dim1];
        let strides = compute_strides(&new_shape);

        Tensor {
            shape: new_shape,
            strides,
            data: new_data,
            device: ConstDevice,
        }
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
const fn compute_strides<const RANK: usize>(shape: &[usize; RANK]) -> [usize; RANK] {
    let mut strides = [0; RANK];
    let mut stride = 1;
    let mut i = RANK;
    while i > 0 {
        i -= 1;
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
    ///
    /// The shape is an array of length `RANK` representing the size of each dimension.
    pub const fn shape(&self) -> &[usize; RANK] {
        &self.shape
    }

    /// Returns the strides of the tensor.
    ///
    /// Strides represent the number of elements to skip in memory to move to the next element
    /// along a specific dimension.
    pub const fn strides(&self) -> &[usize; RANK] {
        &self.strides
    }

    /// Returns a reference to the underlying data as a slice.
    pub fn data(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Returns a mutable reference to the underlying data as a slice.
    ///
    /// # Warning
    ///
    /// Modifying the data directly can be dangerous if you violate invariants (though `Tensor`
    /// doesn't have many invariants on the values themselves). Use with caution.
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// This is equal to the product of the dimensions in the shape.
    pub const fn size(&self) -> usize {
        let mut size = 1;
        let mut i = 0;
        while i < RANK {
            size *= self.shape[i];
            i += 1;
        }
        size
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

    #[test]
    fn test_tensor_accessors() {
        let mut t = Tensor::<f32, 2>::zeros([2, 3]);
        assert_eq!(t.size(), 6);
        assert_eq!(t.strides(), &[3, 1]);

        // Test data_mut
        {
            let data = t.data_mut();
            data[0] = 1.0;
        }
        assert_eq!(t.data()[0], 1.0);
    }

    #[test]
    fn test_compute_strides() {
        let shape = [2, 3, 4];
        let strides = compute_strides(&shape);
        // Stride for dim 2 (last) is 1
        // Stride for dim 1 is 4
        // Stride for dim 0 is 3 * 4 = 12
        assert_eq!(strides, [12, 4, 1]);
    }

    #[test]
    fn test_tensor_error_display() {
        let err = TensorError::ShapeMismatch {
            expected: vec![2, 2],
            got: vec![4],
        };
        assert_eq!(
            format!("{}", err),
            "Shape mismatch: expected [2, 2], got [4]"
        );

        let err = TensorError::IndexOutOfBounds {
            index: vec![3],
            shape: vec![2],
        };
        assert_eq!(
            format!("{}", err),
            "Index out of bounds: index [3] for shape [2]"
        );

        let err = TensorError::Unsupported("foo".to_string());
        assert_eq!(format!("{}", err), "Unsupported operation: foo");
    }

    #[test]
    fn test_tensor_error_debug_fmt() {
        let err = TensorError::Unsupported("foo".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Unsupported"));
    }

    #[test]
    fn test_tensor_clone() {
        let t = Tensor::<f32, 1>::new(vec![1.0, 2.0], [2]).unwrap();
        let t2 = t.clone();
        assert_eq!(t.data(), t2.data());
        assert_eq!(t.shape(), t2.shape());
    }

    #[test]
    fn test_tensor_debug() {
        let t = Tensor::<f32, 1>::new(vec![1.0], [1]).unwrap();
        let debug_str = format!("{:?}", t);
        assert!(debug_str.contains("Tensor"));
        assert!(debug_str.contains("shape"));
        assert!(debug_str.contains("device"));
        assert!(debug_str.contains("CPU"));
    }

    #[test]
    fn test_const_tensor_debug() {
        let data: [f32; 1] = [1.0];
        let t: Tensor<f32, 1, ConstDevice<1>> = Tensor::new_const(data, [1]);
        let debug_str = format!("{:?}", t);
        assert!(debug_str.contains("Tensor"));
        assert!(debug_str.contains("ConstDevice"));
    }
    #[test]
    fn test_const_matmul() {
        // A: [2, 3]
        // 1 2 3
        // 4 5 6
        let a: Tensor<f32, 2, ConstDevice<6>> =
            Tensor::new_const([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

        // B: [3, 2]
        // 7 8
        // 9 10
        // 11 12
        let b: Tensor<f32, 2, ConstDevice<6>> =
            Tensor::new_const([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2]);

        // C = A @ B: [2, 2]
        // 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        // 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        let c: Tensor<f32, 2, ConstDevice<4>> = a.matmul(&b);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    }
    #[test]
    #[should_panic(expected = "Shape mismatch: data length does not match shape product")]
    fn test_const_new_panic() {
        let data: [f32; 3] = [1.0, 2.0, 3.0];
        let _t: Tensor<f32, 2, ConstDevice<3>> = Tensor::new_const(data, [2, 2]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch: new shape size does not match tensor size")]
    fn test_const_reshape_panic() {
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let t: Tensor<f32, 2, ConstDevice<4>> = Tensor::new_const(data, [2, 2]);
        let _t2: Tensor<f32, 1, ConstDevice<4>> = t.reshape([3]);
    }

    #[test]
    #[should_panic(expected = "Transpose requires rank >= 2")]
    fn test_const_transpose_panic() {
        let data: [f32; 2] = [1.0, 2.0];
        let t: Tensor<f32, 1, ConstDevice<2>> = Tensor::new_const(data, [2]);
        let _t2: Tensor<f32, 1, ConstDevice<2>> = t.transpose();
    }

    #[test]
    #[should_panic(expected = "Shape mismatch for matmul")]
    fn test_const_matmul_panic_shape() {
        let a: Tensor<f32, 2, ConstDevice<4>> = Tensor::new_const([1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b: Tensor<f32, 2, ConstDevice<6>> =
            Tensor::new_const([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]); // 2x2 @ 3x2 -> mismatch
        let _c: Tensor<f32, 2, ConstDevice<4>> = a.matmul(&b);
    }

    #[test]
    #[should_panic(expected = "Output size mismatch")]
    fn test_const_matmul_panic_output() {
        let a: Tensor<f32, 2, ConstDevice<4>> = Tensor::new_const([1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b: Tensor<f32, 2, ConstDevice<4>> = Tensor::new_const([1.0, 2.0, 3.0, 4.0], [2, 2]);
        let _c: Tensor<f32, 2, ConstDevice<5>> = a.matmul(&b); // Output should be 4
    }
    #[test]
    fn test_const_tensor() {
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let shape: [usize; 2] = [2, 2];
        let t: Tensor<f32, 2, ConstDevice<4>> = Tensor::new_const(data, shape);

        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.size(), 4);
        assert_eq!(t.data(), &data);
        assert_eq!(t.strides(), &[2, 1]);
    }

    #[test]
    fn test_const_ops() {
        let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: Tensor<f32, 2, ConstDevice<6>> = Tensor::new_const(data, [2, 3]);

        // Test Reshape
        let t_reshaped: Tensor<f32, 3, ConstDevice<6>> = t.reshape([3, 2, 1]);
        assert_eq!(t_reshaped.shape(), &[3, 2, 1]);
        assert_eq!(t_reshaped.data(), &data);

        // Test Transpose
        // [1 2 3]
        // [4 5 6]
        // ->
        // [1 4]
        // [2 5]
        // [3 6]
        let t_transposed: Tensor<f32, 2, ConstDevice<6>> = t.transpose();
        assert_eq!(t_transposed.shape(), &[3, 2]);
        assert_eq!(t_transposed.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_const_compile_check() {
        const DATA: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        const T: Tensor<f32, 2, ConstDevice<4>> = Tensor::new_const(DATA, [2, 2]);
        const T_T: Tensor<f32, 2, ConstDevice<4>> = T.transpose();
        assert_eq!(T_T.data(), &[1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_conflict_check() {
        // This test checks if we can call methods on ConstDevice in a non-const context.
        // If the generic impl applies, it might conflict or fail.
        let data = [1.0, 2.0, 3.0, 4.0];
        let t = Tensor::new_const(data, [2, 2]);

        // This calls the const implementation (inherent method)
        let t_t = t.transpose();
        assert_eq!(t_t.data(), &[1.0, 3.0, 2.0, 4.0]);

        // If we try to use a method that is ONLY in the generic impl but fails for ConstDevice?
        // e.g. matmul calls cpu_matmul which returns Vec.
        // let t2 = t.matmul(&t);
        // This would likely fail to compile if instantiated.
    }
    #[test]
    fn test_generic_transpose() {
        use crate::tensor::ops::TensorOps;

        fn generic_transpose<T: TensorElem, const RANK: usize, D: Device>(
            t: &Tensor<T, RANK, D>,
        ) -> crate::tensor::Result<Tensor<T, RANK, D>>
        where
            Tensor<T, RANK, D>: TensorOps<T, RANK, Device = D>,
        {
            t.transpose()
        }

        // Test CPU
        let t_cpu: Tensor<f32, 2> =
            Tensor::<f32, 2>::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let t_cpu_res = generic_transpose(&t_cpu);
        let t_cpu_t = t_cpu_res.unwrap();
        assert_eq!(t_cpu_t.shape(), &[2, 2]);
        assert_eq!(t_cpu_t.data(), &[1.0, 3.0, 2.0, 4.0]);

        // Test ConstDevice
        let t_const = Tensor::new_const([1.0, 2.0, 3.0, 4.0], [2, 2]);
        let t_const_res = generic_transpose(&t_const);
        let t_const_t = t_const_res.unwrap();
        assert_eq!(t_const_t.shape(), &[2, 2]);
        assert_eq!(t_const_t.data(), &[1.0, 3.0, 2.0, 4.0]);
    }
}
