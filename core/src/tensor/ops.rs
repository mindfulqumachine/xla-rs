//! Tensor operations.
//!
//! # Overview
//!
//! This module implements the mathematical engine of `xla-rs`. It handles:
//! - **Element-wise Arithmetic**: `+`, `-`, `*`, `/` (with broadcasting).
//! - **Matrix Multiplication**: Dot products and batched matmuls.
//! - **Parallelism**: Uses `rayon` to parallelize operations across CPU cores.
//!
//! # Broadcasting
//!
//! Broadcasting is a powerful mechanism that allows operations on tensors of different shapes.
//! For example, you can add a 1D bias vector to a 2D weight matrix.
//!
//! **Rules:**
//! 1. Dimensions are aligned from the right (last dimension).
//! 2. Two dimensions are compatible if:
//!    - They are equal, OR
//!    - One of them is 1.
//!
//! > [!NOTE]
//! > Currently, `xla-rs` implements strict shape checking for simplicity. Full broadcasting
//! > (expanding dimensions of size 1) is a planned feature. For now, shapes must match exactly
//! > for element-wise operations.
//!
//! # Parallelism
//!
//! > [!TIP]
//! > **Efficiency Note**: Operations are parallelized using `rayon`. This means that for small tensors,
//! > the overhead of thread synchronization might outweigh the benefits. `xla-rs` is optimized for
//! > "medium-sized" tensors typical in LLM inference (e.g., hidden size 768+).
//!
//! # Examples
//!
//! ```rust
//! use xla_rs::tensor::Tensor;
//!
//! let a = Tensor::<f32, 1>::new(vec![1.0, 2.0], [2]).unwrap();
//! let b = Tensor::<f32, 1>::new(vec![3.0, 4.0], [2]).unwrap();
//!
//! // Element-wise addition
//! let c = (&a + &b).unwrap();
//! assert_eq!(c.data(), &[4.0, 6.0]);
//! ```

use super::{Cpu, Device, Result, Tensor, TensorElem, TensorError};

use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

/// Implements a binary arithmetic operation trait (e.g., `Add`, `Sub`) for `&Tensor`.
///
/// This macro handles the boilerplate of:
/// 1. Checking shape compatibility.
/// 2. Creating a new output tensor.
/// 3. Performing the element-wise operation in parallel using `rayon`.
///
/// # Arguments
///
/// * `$trait` - The trait to implement (e.g., `Add`).
/// * `$method` - The method name of the trait (e.g., `add`).
macro_rules! impl_bin_op {
    ($trait:ident, $method:ident) => {
        impl<T, const RANK: usize> $trait for &Tensor<T, RANK, Cpu>
        where
            T: TensorElem,
        {
            type Output = crate::tensor::Result<Tensor<T, RANK, Cpu>>;

            fn $method(self, rhs: Self) -> Self::Output {
                if self.shape != rhs.shape {
                    return Err(TensorError::ShapeMismatch {
                        expected: self.shape.to_vec(),
                        got: rhs.shape.to_vec(),
                    });
                }

                let mut out = Tensor::zeros(self.shape);
                // Seamless parallelism using rayon
                out.data
                    .as_mut_slice()
                    .par_iter_mut()
                    .zip(self.data.as_slice().par_iter())
                    .zip(rhs.data.as_slice().par_iter())
                    .for_each(|((o, a), b)| {
                        *o = a.$method(*b);
                    });

                Ok(out)
            }
        }
    };
}

impl_bin_op!(Add, add);
impl_bin_op!(Sub, sub);
impl_bin_op!(Mul, mul);
impl_bin_op!(Div, div);

/// Trait for Tensor operations that depend on the device implementation.
///
/// This allows generic code to use operations like `transpose` regardless of the device.
pub trait TensorOps<T: TensorElem, const RANK: usize> {
    /// Transposes the tensor.
    fn transpose(&self) -> Result<Tensor<T, RANK, <Self as HasDevice>::Device>>
    where
        Self: HasDevice,
        <Self as HasDevice>::Device: Device;

    type Device;
}

/// Helper trait to access the device type.
pub trait HasDevice {
    type Device;
}

impl<T: TensorElem, const RANK: usize, D: Device> HasDevice for Tensor<T, RANK, D> {
    type Device = D;
}

impl<T, const RANK: usize, D: Device> TensorOps<T, RANK> for Tensor<T, RANK, D>
where
    T: TensorElem,
{
    type Device = D;

    fn transpose(&self) -> Result<Tensor<T, RANK, <Self as HasDevice>::Device>> {
        let out_data = D::transpose(&self.data, &self.shape)?;

        let mut new_shape = self.shape;
        if RANK >= 2 {
            new_shape.swap(RANK - 1, RANK - 2);
        }

        let strides = crate::tensor::compute_strides(&new_shape);
        Ok(Tensor {
            shape: new_shape,
            strides,
            data: out_data,
            device: self.device.clone(),
        })
    }
}

impl<T, const RANK: usize> Tensor<T, RANK, Cpu>
where
    T: TensorElem,
{
    /// Applies a function element-wise to the tensor.
    ///
    /// Creates a new tensor with the same shape, where each element is the result of applying
    /// the closure `f` to the corresponding element in the input tensor.
    ///
    /// # Arguments
    ///
    /// * `f` - A closure that takes an element of type `T` and returns a value of type `T`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use xla_rs::tensor::Tensor;
    /// let t = Tensor::<f32, 1>::new(vec![1.0, 2.0, 3.0], [3]).unwrap();
    /// let squared = t.map(|x| x * x);
    /// assert_eq!(squared.data(), &[1.0, 4.0, 9.0]);
    /// ```
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T + Sync + Send,
    {
        let mut out = Tensor::zeros(self.shape);
        out.data
            .as_mut_slice()
            .par_iter_mut()
            .zip(self.data.as_slice().par_iter())
            .for_each(|(o, i)| *o = f(*i));
        out
    }

    /// Matrix Multiplication.
    /// Supports:
    /// - 2D x 2D: [M, K] x [K, N] -> [M, N]
    /// - 3D x 3D: [B, M, K] x [B, K, N] -> [B, M, N] (Batched Matmul)
    ///
    /// Performs matrix multiplication on the last two dimensions of the tensors.
    /// If the rank is greater than 2, the leading dimensions are treated as batch dimensions.
    /// This is known as **Batched Matrix Multiplication**.
    ///
    /// # Mathematical Definition
    ///
    /// For tensors of rank $N$, this operation corresponds to the Einstein summation:
    /// `...mk,...kn->...mn`
    ///
    /// # Examples
    ///
    /// - **Rank 2 (Matrix x Matrix)**: `[M, K] x [K, N] -> [M, N]`
    /// - **Rank 3 (Batch x Matrix x Matrix)**: `[B, M, K] x [B, K, N] -> [B, M, N]`
    /// - **Rank 4**: `[B, H, M, K] x [B, H, K, N] -> [B, H, M, N]`
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        // Compile-time check
        const { assert!(RANK >= 2, "Matmul requires rank >= 2") };
        self.matmul_impl(rhs)
    }

    /// Internal implementation of Matrix Multiplication.
    fn matmul_impl(&self, rhs: &Self) -> Result<Self> {
        let m = self.shape[RANK - 2];
        let _k = self.shape[RANK - 1];
        let _k2 = rhs.shape[RANK - 2];
        let n = rhs.shape[RANK - 1];

        // Check batch dimensions
        if self.shape[..RANK - 2] != rhs.shape[..RANK - 2] {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.to_vec(),
                got: rhs.shape.to_vec(),
            });
        }

        let mut out_shape = self.shape;
        out_shape[RANK - 2] = m;
        out_shape[RANK - 1] = n;

        // Delegate to the kernel
        // This is where you would swap in a BLAS call or other accelerator
        let out_data = xla_rs_kernels::cpu_matmul(
            self.data.as_slice(),
            rhs.data.as_slice(),
            &self.shape,
            &rhs.shape,
        )
        .map_err(|e| match e {
            xla_rs_kernels::KernelError::ShapeMismatch { expected, got } => {
                TensorError::ShapeMismatch { expected, got }
            }
        })?;

        let strides = crate::tensor::compute_strides(&out_shape);
        Ok(Tensor {
            shape: out_shape,
            strides,
            data: out_data,
            device: Cpu,
        })
    }

    /// Performs 2D Convolution.
    ///
    /// # Arguments
    ///
    /// * `weight` - Weight tensor of shape `[out_channels, in_channels, kernel_h, kernel_w]`.
    /// * `stride` - Stride of the convolution `[stride_h, stride_w]`.
    /// * `padding` - Padding added to both sides of the input `[pad_h, pad_w]`.
    /// * `dilation` - Dilation of the kernel `[dil_h, dil_w]`.
    ///
    /// # Returns
    ///
    /// A new tensor resulting from the convolution.
    /// Output shape: `[batch_size, out_channels, out_h, out_w]`
    pub fn conv2d(
        &self,
        weight: &Tensor<T, 4, Cpu>,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Tensor<T, 4, Cpu>> {
        // Compile-time check
        const { assert!(RANK == 4, "Conv2d requires rank 4 input") };

        let out_data = xla_rs_kernels::cpu_conv2d(
            self.data.as_slice(),
            weight.data.as_slice(),
            &self.shape,
            &weight.shape,
            stride,
            padding,
            dilation,
        )
        .map_err(|e| match e {
            xla_rs_kernels::KernelError::ShapeMismatch { expected, got } => {
                TensorError::ShapeMismatch { expected, got }
            }
        })?;

        // Calculate output shape to construct Tensor
        // We can duplicate the logic or trust the kernel returns correct size and we just need to compute shape to store it.
        // Or we can ask kernel to return shape? No, kernel returns Vec<T>.
        // Let's recompute shape here.

        let batch_size = self.shape[0];
        // let in_channels = self.shape[1];
        let in_h = self.shape[2];
        let in_w = self.shape[3];

        let out_channels = weight.shape[0];
        let k_h = weight.shape[2];
        let k_w = weight.shape[3];

        let stride_h = stride[0];
        let stride_w = stride[1];
        let pad_h = padding[0];
        let pad_w = padding[1];
        let dil_h = dilation[0];
        let dil_w = dilation[1];

        let effective_k_h = k_h + (k_h - 1) * (dil_h - 1);
        let effective_k_w = k_w + (k_w - 1) * (dil_w - 1);

        let out_h = (in_h + 2 * pad_h - effective_k_h) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - effective_k_w) / stride_w + 1;

        let out_shape = [batch_size, out_channels, out_h, out_w];
        let strides = crate::tensor::compute_strides(&out_shape);

        Ok(Tensor {
            shape: out_shape,
            strides,
            data: out_data,
            device: Cpu,
        })
    }

    /// Performs 2D Max Pooling.
    ///
    /// # Arguments
    ///
    /// * `kernel_size` - Size of the pooling window `[k_h, k_w]`.
    /// * `stride` - Stride of the pooling `[stride_h, stride_w]`.
    /// * `padding` - Padding added to both sides of the input `[pad_h, pad_w]`.
    ///
    /// # Returns
    ///
    /// A new tensor resulting from the max pooling.
    /// Output shape: `[batch_size, channels, out_h, out_w]`
    pub fn max_pool2d(
        &self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Tensor<T, 4, Cpu>> {
        // Compile-time check
        const { assert!(RANK == 4, "MaxPool2d requires rank 4 input") };

        let out_data = xla_rs_kernels::cpu_max_pool2d(
            self.data.as_slice(),
            &self.shape,
            kernel_size,
            stride,
            padding,
        )
        .map_err(|e| match e {
            xla_rs_kernels::KernelError::ShapeMismatch { expected, got } => {
                TensorError::ShapeMismatch { expected, got }
            }
        })?;

        let batch_size = self.shape[0];
        let channels = self.shape[1];
        let in_h = self.shape[2];
        let in_w = self.shape[3];

        let k_h = kernel_size[0];
        let k_w = kernel_size[1];
        let stride_h = stride[0];
        let stride_w = stride[1];
        let pad_h = padding[0];
        let pad_w = padding[1];

        let out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

        let out_shape = [batch_size, channels, out_h, out_w];
        let strides = crate::tensor::compute_strides(&out_shape);

        Ok(Tensor {
            shape: out_shape,
            strides,
            data: out_data,
            device: Cpu,
        })
    }

    /// Transposes two specific axes of the tensor.
    ///
    /// This operation creates a new tensor with the data physically permuted to match the new shape.
    ///
    /// # Arguments
    ///
    /// * `ax1` - The first axis to swap.
    /// * `ax2` - The second axis to swap.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::IndexOutOfBounds` if `ax1` or `ax2` are out of bounds.
    /// Returns `TensorError::Unsupported` for complex permutations not yet optimized.
    pub fn transpose_axes(&self, ax1: usize, ax2: usize) -> Result<Self> {
        if ax1 >= RANK || ax2 >= RANK {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![ax1, ax2],
                shape: self.shape.to_vec(),
            });
        }
        if ax1 == ax2 {
            return Ok(self.clone());
        }

        let mut new_shape = self.shape;
        new_shape.swap(ax1, ax2);

        // Compute new strides based on old strides
        // But since we own data, we can't just change strides without permuting data if we want it contiguous.
        // My Tensor struct enforces contiguous storage (Vec<T>).
        // So we must physically move data.

        let mut out = Tensor::zeros(new_shape);

        // Generic permute is hard. Implementing for specific cases used in Attention.
        // Case: [B, S, H, D] -> [B, H, S, D] (swap 1 and 2) where RANK=4.

        if RANK == 4 {
            // Generic Rank 4 transpose
            // We want to iterate over the *output* tensor linearly or in chunks, and map to *input* tensor.
            // Or iterate over input dimensions in the new order.

            let out_ptr = out.data.as_mut_slice();
            let in_ptr = self.data.as_slice();
            let in_strides = self.strides;

            // New shape is `new_shape`.
            // We can iterate over the 4 dimensions of the OUTPUT.
            // Let's call them d0, d1, d2, d3.
            let _d0 = new_shape[0];
            let d1 = new_shape[1];
            let d2 = new_shape[2];
            let d3 = new_shape[3];

            // We need to map output indices (i0, i1, i2, i3) to input indices.
            // If we swapped ax1 and ax2:
            // input_indices[ax1] = output_indices[ax2]
            // input_indices[ax2] = output_indices[ax1]
            // others are same.

            // Let's pre-calculate the stride mapping.
            // input_offset = i0 * S0 + i1 * S1 + i2 * S2 + i3 * S3
            // where S_k is the stride of the k-th dimension in the INPUT.
            // But we are iterating i0..i3 which are OUTPUT dimensions.
            // So we need to know which Input Stride corresponds to Output Dimension k.
            // If we swapped ax1 and ax2, then:
            // Output Dim ax1 corresponds to Input Dim ax2.
            // Output Dim ax2 corresponds to Input Dim ax1.
            // Other Output Dim k corresponds to Input Dim k.

            let mut mapped_strides = [0; 4];
            for i in 0..4 {
                if i == ax1 {
                    mapped_strides[i] = in_strides[ax2];
                } else if i == ax2 {
                    mapped_strides[i] = in_strides[ax1];
                } else {
                    mapped_strides[i] = in_strides[i];
                }
            }

            let s0 = mapped_strides[0];
            let s1 = mapped_strides[1];
            let s2 = mapped_strides[2];
            let s3 = mapped_strides[3];

            // Parallelize over d0 (Batch) and d1
            // Flatten d0 and d1 for better parallelism if d0 is small?
            // Or just d0.

            out_ptr
                .par_chunks_mut(d1 * d2 * d3)
                .enumerate()
                .for_each(|(i0, chunk)| {
                    // i0 is the index for dimension 0
                    for i1 in 0..d1 {
                        for i2 in 0..d2 {
                            for i3 in 0..d3 {
                                // Output linear index within chunk
                                let out_idx = i1 * (d2 * d3) + i2 * d3 + i3;

                                // Input linear index
                                let in_idx = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3;

                                chunk[out_idx] = in_ptr[in_idx];
                            }
                        }
                    }
                });

            Ok(out)
        } else if RANK == 3 {
            // Generic Rank 3 transpose
            let out_ptr = out.data.as_mut_slice();
            let in_ptr = self.data.as_slice();
            let in_strides = self.strides;

            let _d0 = new_shape[0];
            let d1 = new_shape[1];
            let d2 = new_shape[2];

            let mut mapped_strides = [0; 3];
            for i in 0..3 {
                if i == ax1 {
                    mapped_strides[i] = in_strides[ax2];
                } else if i == ax2 {
                    mapped_strides[i] = in_strides[ax1];
                } else {
                    mapped_strides[i] = in_strides[i];
                }
            }

            let s0 = mapped_strides[0];
            let s1 = mapped_strides[1];
            let s2 = mapped_strides[2];

            out_ptr
                .par_chunks_mut(d1 * d2)
                .enumerate()
                .for_each(|(i0, chunk)| {
                    for i1 in 0..d1 {
                        for i2 in 0..d2 {
                            let out_idx = i1 * d2 + i2;
                            let in_idx = i0 * s0 + i1 * s1 + i2 * s2;
                            chunk[out_idx] = in_ptr[in_idx];
                        }
                    }
                });
            Ok(out)
        } else {
            Err(TensorError::Unsupported(format!(
                "General transpose_axes not impl for rank {} axes {},{}",
                RANK, ax1, ax2
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic() {
        let a = Tensor::<f32, 1>::new(vec![1.0, 2.0], [2]).unwrap();
        let b = Tensor::<f32, 1>::new(vec![3.0, 4.0], [2]).unwrap();

        // Add
        let c = (&a + &b).unwrap();
        assert_eq!(c.data(), &[4.0, 6.0]);

        // Sub
        let c = (&a - &b).unwrap();
        assert_eq!(c.data(), &[-2.0, -2.0]);

        // Mul
        let d = (&a * &b).unwrap();
        assert_eq!(d.data(), &[3.0, 8.0]);

        // Div
        let d = (&a / &b).unwrap();
        assert_eq!(d.data(), &[1.0 / 3.0, 2.0 / 4.0]);

        // Mismatch
        let _e = Tensor::<f32, 1>::new(vec![1.0], [1]).unwrap();
        let f = Tensor::<f32, 1>::new(vec![1.0, 2.0, 3.0], [3]).unwrap();
        let err = &a + &f;
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));

        let err = &a - &f;
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));

        let err = &a * &f;
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));

        let err = &a / &f;
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_map() {
        let a = Tensor::<f32, 1>::new(vec![1.0, 2.0, 3.0], [3]).unwrap();
        let b = a.map(|x| x * 2.0);
        assert_eq!(b.data(), &[2.0, 4.0, 6.0]);
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
    fn test_matmul_3d() {
        // Batched Matmul: [B, M, K] x [B, K, N] -> [B, M, N]
        // B=2, M=2, K=2, N=2
        // Batch 1: Identity * Identity = Identity
        // Batch 2: 2*Identity * 3*Identity = 6*Identity

        let batch1_a = vec![1.0, 0.0, 0.0, 1.0]; // Identity
        let batch2_a = vec![2.0, 0.0, 0.0, 2.0]; // 2*Identity
        let mut a_data = batch1_a;
        a_data.extend(batch2_a);
        let a = Tensor::<f32, 3>::new(a_data, [2, 2, 2]).unwrap();

        let batch1_b = vec![1.0, 0.0, 0.0, 1.0]; // Identity
        let batch2_b = vec![3.0, 0.0, 0.0, 3.0]; // 3*Identity
        let mut b_data = batch1_b;
        b_data.extend(batch2_b);
        let b = Tensor::<f32, 3>::new(b_data, [2, 2, 2]).unwrap();

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);

        let expected_batch1 = vec![1.0, 0.0, 0.0, 1.0];
        let expected_batch2 = vec![6.0, 0.0, 0.0, 6.0];
        let mut expected = expected_batch1;
        expected.extend(expected_batch2);

        assert_eq!(c.data(), &expected[..]);
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
    fn test_matmul_4d() {
        // [B, S, M, K] x [B, S, K, N] -> [B, S, M, N]
        // Shape: [1, 2, 2, 2] x [1, 2, 2, 2] -> [1, 2, 2, 2]
        // Batch 1, Seq 1: Identity * Identity = Identity
        // Batch 1, Seq 2: 2*Identity * 3*Identity = 6*Identity

        let s1_a = vec![1.0, 0.0, 0.0, 1.0];
        let s2_a = vec![2.0, 0.0, 0.0, 2.0];
        let mut a_data = s1_a;
        a_data.extend(s2_a);
        let a = Tensor::<f32, 4>::new(a_data, [1, 2, 2, 2]).unwrap();

        let s1_b = vec![1.0, 0.0, 0.0, 1.0];
        let s2_b = vec![3.0, 0.0, 0.0, 3.0];
        let mut b_data = s1_b;
        b_data.extend(s2_b);
        let b = Tensor::<f32, 4>::new(b_data, [1, 2, 2, 2]).unwrap();

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[1, 2, 2, 2]);

        let expected_s1 = vec![1.0, 0.0, 0.0, 1.0];
        let expected_s2 = vec![6.0, 0.0, 0.0, 6.0];
        let mut expected = expected_s1;
        expected.extend(expected_s2);

        assert_eq!(c.data(), &expected[..]);
    }

    #[test]
    fn test_matmul_10d() {
        // 10D Tensor
        // Shape: [1, ..., 1, 2, 2] (8 ones, then 2, 2)
        let shape = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0]; // [1 2; 3 4]
        let a = Tensor::<f32, 10>::new(data.clone(), shape).unwrap();
        let b = Tensor::<f32, 10>::new(data, shape).unwrap();

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &shape);

        // [1 2; 3 4] * [1 2; 3 4] = [7 10; 15 22]
        assert_eq!(c.data(), &[7.0, 10.0, 15.0, 22.0]);
    }

    #[test]
    fn test_transpose_4d() {
        // [B, S, M, N] -> [B, S, N, M]
        // [1, 2, 2, 3] -> [1, 2, 3, 2]
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = Tensor::<f32, 4>::new(data, [1, 2, 2, 3]).unwrap();

        let t_t = t.transpose().unwrap();
        assert_eq!(t_t.shape(), &[1, 2, 3, 2]);

        // First matrix (0..6):
        // [0 1 2]
        // [3 4 5]
        // Transpose:
        // [0 3]
        // [1 4]
        // [2 5]
        // -> 0, 3, 1, 4, 2, 5

        // Second matrix (6..12):
        // [6 7 8]
        // [9 10 11]
        // Transpose:
        // [6 9]
        // [7 10]
        // [8 11]
        // -> 6, 9, 7, 10, 8, 11

        let expected = vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 6.0, 9.0, 7.0, 10.0, 8.0, 11.0];
        assert_eq!(t_t.data(), &expected[..]);
    }

    #[test]
    fn test_transpose_10d() {
        // 10D Tensor
        // Shape: [1, ..., 1, 2, 3]
        let shape = [1, 1, 1, 1, 1, 1, 1, 1, 2, 3];
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let t = Tensor::<f32, 10>::new(data, shape).unwrap();

        let t_t = t.transpose().unwrap();
        let expected_shape = [1, 1, 1, 1, 1, 1, 1, 1, 3, 2];
        assert_eq!(t_t.shape(), &expected_shape);

        // [0 1 2]
        // [3 4 5]
        // ->
        // [0 3]
        // [1 4]
        // [2 5]
        assert_eq!(t_t.data(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn test_transpose_error() {
        let t = Tensor::<f32, 1>::new(vec![1.0, 2.0], [2]).unwrap();
        let err = t.transpose();
        assert!(matches!(err, Err(TensorError::Unsupported(_))));
    }

    #[test]
    fn test_transpose_axes_error() {
        let t = Tensor::<f32, 2>::zeros([2, 2]);
        let err = t.transpose_axes(0, 2); // 2 is out of bounds
        assert!(matches!(err, Err(TensorError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_transpose_axes_identity() {
        let t = Tensor::<f32, 2>::zeros([2, 2]);
        let t2 = t.transpose_axes(0, 0).unwrap();
        assert_eq!(t.shape(), t2.shape());
    }

    #[test]
    fn test_transpose_axes_unsupported() {
        let t = Tensor::<f32, 3>::zeros([2, 2, 2]);
        // 3D transpose axes not fully implemented in the simplified logic
        let err = t.transpose_axes(0, 1);
        assert!(matches!(err, Err(TensorError::Unsupported(_))));
    }

    #[test]
    fn test_matmul_batch_mismatch() {
        let a = Tensor::<f32, 3>::zeros([2, 2, 2]);
        let b = Tensor::<f32, 3>::zeros([3, 2, 2]); // Batch size 3 vs 2
        let err = a.matmul(&b);
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
    }
}
