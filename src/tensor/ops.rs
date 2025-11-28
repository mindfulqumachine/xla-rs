//! Tensor operations.
//!
//! This module implements various mathematical operations for Tensors, including
//! element-wise arithmetic (Add, Sub, Mul, Div), matrix multiplication, and broadcasting.
//! It uses `rayon` for parallel execution on the CPU.

use super::{Cpu, Result, Tensor, TensorElem, TensorError};
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

/// Implements a binary arithmetic operation trait (e.g., `Add`, `Sub`) for `&Tensor`.
///
/// This macro handles the boilerplate of checking shape compatibility, creating a new
/// output tensor, and performing the element-wise operation in parallel using `rayon`.
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
            type Output = Result<Tensor<T, RANK, Cpu>>;

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

impl<T, const RANK: usize> Tensor<T, RANK, Cpu>
where
    T: TensorElem,
{
    /// Applies a function element-wise.
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
    /// Matrix Multiplication.
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

    fn matmul_impl(&self, rhs: &Self) -> Result<Self> {
        let m = self.shape[RANK - 2];
        let k = self.shape[RANK - 1];
        let k2 = rhs.shape[RANK - 2];
        let n = rhs.shape[RANK - 1];

        if k != k2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![m, k],
                got: vec![k2, n],
            });
        }

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

        let mut out = Tensor::zeros(out_shape);

        // Optimization: Transpose rhs to allow sequential access (cache friendly)
        // rhs_t shape: [..., N, K] (swapped last two dims)
        let rhs_t = rhs.transpose()?;

        // Parallelize over rows of the output matrices across all batches
        // Output shape is [Batch..., M, N]
        // We iterate over (Batch... * M) rows, each of size N

        out.data
            .as_mut_slice()
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(global_row_idx, out_row)| {
                let batch_idx = global_row_idx / m;
                let row_in_matrix = global_row_idx % m;

                // Calculate offsets for input tensors
                // Assumes contiguous memory layout (which Tensor enforces)
                let a_batch_offset = batch_idx * m * k;
                // rhs_t has shape [..., N, K], so batch offset is batch_idx * N * K
                let b_t_batch_offset = batch_idx * n * k;

                let a_row_start = a_batch_offset + row_in_matrix * k;
                let a_slice = &self.data.as_slice()[a_row_start..a_row_start + k];

                for (col_in_matrix, out_elem) in out_row.iter_mut().enumerate() {
                    // We want dot product of:
                    // A row: `row_in_matrix`
                    // B col: `col_in_matrix` -> which is rhs_t row `col_in_matrix`

                    let b_t_row_start = b_t_batch_offset + col_in_matrix * k;
                    let b_t_slice = &rhs_t.data.as_slice()[b_t_row_start..b_t_row_start + k];

                    let mut sum = T::zero();
                    // Vectorizable loop
                    for (&val_a, &val_b) in a_slice.iter().zip(b_t_slice.iter()) {
                        sum += val_a * val_b;
                    }
                    *out_elem = sum;
                }
            });
        Ok(out)
    }

    /// Transposes the tensor.
    ///
    /// Swaps the last two dimensions.
    /// - For 2D tensors (matrices), this is a standard matrix transpose.
    /// - For N-D tensors, it swaps the last two axes (e.g., [B, M, N] -> [B, N, M]).
    ///
    /// # Errors
    ///
    /// Returns `TensorError::Unsupported` if the tensor rank is less than 2.
    pub fn transpose(&self) -> Result<Self> {
        if RANK < 2 {
            return Err(TensorError::Unsupported(
                "Transpose requires rank >= 2".into(),
            ));
        }

        let mut new_shape = self.shape;
        new_shape.swap(RANK - 1, RANK - 2);

        let mut out = Tensor::zeros(new_shape);

        // Dimensions of the inner matrix
        let m = self.shape[RANK - 2];
        let n = self.shape[RANK - 1];

        // Flatten batch dimensions
        // The total number of matrices to transpose is the product of all dimensions except the last two.
        // If RANK=2, batch_size=1.
        // let batch_size: usize = self.shape[..RANK - 2].iter().product();
        // Note: product of empty slice is 1, which is correct for Rank 2.

        // We parallelize over the rows of the OUTPUT tensor.
        // The output tensor has shape [Batch..., N, M].
        // So we view it as `batch_size * N` rows, each of length `M`.
        out.data
            .as_mut_slice()
            .par_chunks_mut(m)
            .enumerate()
            .for_each(|(i, out_row)| {
                // `i` is the global row index in the flattened output [Batch * N, M]
                let batch_idx = i / n;
                let col_idx = i % n; // This corresponds to the column index in the input matrix

                // Calculate the base offset for this batch in the input data
                let input_batch_offset = batch_idx * m * n;

                // Copy the column `col_idx` from the input matrix to `out_row`
                for r in 0..m {
                    // Input is [M, N]. We want element at (r, col_idx).
                    // Index = input_batch_offset + r * N + col_idx
                    out_row[r] = self.data.as_slice()[input_batch_offset + r * n + col_idx];
                }
            });

        Ok(out)
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

        if RANK == 4 && ((ax1 == 1 && ax2 == 2) || (ax1 == 2 && ax2 == 1)) {
            let [_, _, _, _] = self.shape[0..4].try_into().unwrap();
            // Input: B, S, H, D. Output: B, H, S, D.
            // We want to write to [b, h, s, d].
            // Iterate output
            let out_s = out.strides;
            let in_s = self.strides;

            // We can use a generic loop with recursion or specific nested loops.
            // Nested loops for 4D.
            // Parallelize on B.

            // Check if ax1/ax2 match expectation.
            // The shape in `self` is [B, S, H, D] (if coming from reshape).
            // If we swap 1 and 2, we get [B, H, S, D].

            // But let's use the strides to be generic for 4D.

            let out_ptr = out.data.as_mut_slice();
            let in_ptr = self.data.as_slice();

            // Parallelize outer dim
            out_ptr
                .par_chunks_mut(out_s[0])
                .enumerate()
                .for_each(|(i, batch_chunk)| {
                    // i is batch index
                    // batch_chunk is [H, S, D] size flattened
                    // We need to iterate over H, S, D
                    let new_dim1 = new_shape[1]; // H
                    let new_dim2 = new_shape[2]; // S
                    let new_dim3 = new_shape[3]; // D

                    for j in 0..new_dim1 {
                        for k in 0..new_dim2 {
                            for l in 0..new_dim3 {
                                // Output index: i, j, k, l (linear: i*s0 + j*s1 + k*s2 + l*s3)
                                // This chunk corresponds to `i`. so offset is j*out_s[1] + ...
                                let out_idx = j * out_s[1] + k * out_s[2] + l * out_s[3];

                                // Input index: i, k, j, l (swapped j and k compared to output dims mapping)
                                // Wait, we swapped axes.
                                // Input indices:
                                // idx[ax1] = out_idx[ax2]
                                // idx[ax2] = out_idx[ax1]
                                // indices: [i, k, j, l] if we swapped 1 and 2.
                                // Input stride:
                                let in_idx = i * in_s[0] + k * in_s[1] + j * in_s[2] + l * in_s[3];

                                batch_chunk[out_idx] = in_ptr[in_idx];
                            }
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
}
