use super::{Tensor, TensorElem, TensorError, Device, Cpu, Result};
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub, Div};

// Simple macro to implement arithmetic traits
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
                out.data.as_mut_slice()
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
        out.data.as_mut_slice()
            .par_iter_mut()
            .zip(self.data.as_slice().par_iter())
            .for_each(|(o, i)| *o = f(*i));
        out
    }

    /// Matrix Multiplication.
    /// Supports:
    /// - 2D x 2D: [M, K] x [K, N] -> [M, N]
    /// - 3D x 3D: [B, M, K] x [B, K, N] -> [B, M, N] (Batched Matmul)
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if RANK == 2 {
            self.matmul_impl(rhs)
        } else if RANK == 3 {
            self.matmul_impl(rhs)
        } else {
            Err(TensorError::Unsupported(format!("Matmul not implemented for rank {}", RANK)))
        }
    }

    fn matmul_impl(&self, rhs: &Self) -> Result<Self> {
        if RANK == 2 {
             let [m, k] = self.shape[0..2] else { unreachable!() };
             let [k2, n] = rhs.shape[0..2] else { unreachable!() };

             if k != k2 { return Err(TensorError::ShapeMismatch { expected: vec![m, k], got: vec![k2, n] }); }

             let mut out_shape = [0; RANK];
             out_shape[0] = m;
             out_shape[1] = n;

             let mut out = Tensor::zeros(out_shape);

             out.data.as_mut_slice()
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, out_row)| {
                    for j in 0..n {
                        let mut sum = T::zero();
                        for l in 0..k {
                             let val_a = self.data.as_slice()[i * self.strides[0] + l * self.strides[1]];
                             let val_b = rhs.data.as_slice()[l * rhs.strides[0] + j * rhs.strides[1]];
                             sum += val_a * val_b;
                        }
                        out_row[j] = sum;
                    }
                });
             Ok(out)

        } else if RANK == 3 {
             let [b, m, k] = self.shape[0..3] else { unreachable!() };
             let [b2, k2, n] = rhs.shape[0..3] else { unreachable!() };

             if b != b2 || k != k2 { return Err(TensorError::ShapeMismatch { expected: vec![b, m, k], got: vec![b2, k2, n] }); }

             let mut out_shape = [0; RANK];
             out_shape[0] = b;
             out_shape[1] = m;
             out_shape[2] = n;

             let mut out = Tensor::zeros(out_shape);

             out.data.as_mut_slice()
                .par_chunks_mut(m * n)
                .enumerate()
                .for_each(|(batch_idx, out_matrix)| {
                    let a_offset = batch_idx * m * k;
                    let b_offset = batch_idx * k * n;

                    for r in 0..m {
                        for c in 0..n {
                            let mut sum = T::zero();
                            for l in 0..k {
                                let val_a = self.data.as_slice()[a_offset + r * self.strides[1] + l * self.strides[2]];
                                let val_b = rhs.data.as_slice()[b_offset + l * rhs.strides[1] + c * rhs.strides[2]];
                                sum += val_a * val_b;
                            }
                            out_matrix[r * n + c] = sum;
                        }
                    }
                });
             Ok(out)
        } else {
             Err(TensorError::Unsupported(format!("Matmul not implemented for rank {}", RANK)))
        }
    }

    pub fn transpose(&self) -> Result<Self> {
        if RANK < 2 {
            return Err(TensorError::Unsupported("Transpose requires rank >= 2".into()));
        }

        let mut new_shape = self.shape;
        new_shape.swap(RANK - 1, RANK - 2);

        let mut out = Tensor::zeros(new_shape);

        if RANK == 2 {
             let [m, n] = self.shape[0..2].try_into().unwrap();
             out.data.as_mut_slice().par_chunks_mut(m).enumerate().for_each(|(c, col)| {
                 for r in 0..m {
                     col[r] = self.data.as_slice()[r * n + c];
                 }
             });
        } else if RANK == 3 {
             let [b, m, n] = self.shape[0..3].try_into().unwrap();
             out.data.as_mut_slice().par_chunks_mut(n * m).enumerate().for_each(|(batch_idx, matrix)| {
                 let offset = batch_idx * m * n;
                 for c in 0..n {
                     for r in 0..m {
                         matrix[c * m + r] = self.data.as_slice()[offset + r * n + c];
                     }
                 }
             });
        } else if RANK == 4 {
             let [_b, _h, m, n] = self.shape[0..4].try_into().unwrap();
             out.data.as_mut_slice().par_chunks_mut(n * m).enumerate().for_each(|(idx, matrix)| {
                 let offset = idx * m * n;
                 for c in 0..n {
                     for r in 0..m {
                         matrix[c * m + r] = self.data.as_slice()[offset + r * n + c];
                     }
                 }
             });
        } else {
             return Err(TensorError::Unsupported(format!("Transpose not impl for rank {}", RANK)));
        }

        Ok(out)
    }

    /// Transpose arbitrary dimensions.
    /// Only efficient for contiguous tensors.
    pub fn transpose_axes(&self, ax1: usize, ax2: usize) -> Result<Self> {
        if ax1 >= RANK || ax2 >= RANK {
            return Err(TensorError::IndexOutOfBounds { index: vec![ax1, ax2], shape: self.shape.to_vec() });
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
             let [b, s, h, d] = self.shape[0..4].try_into().unwrap();
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
             out_ptr.par_chunks_mut(out_s[0]).enumerate().for_each(|(i, batch_chunk)| {
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
             Err(TensorError::Unsupported(format!("General transpose_axes not impl for rank {} axes {},{}", RANK, ax1, ax2)))
        }
    }
}
