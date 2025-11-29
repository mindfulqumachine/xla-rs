use crate::{KernelElem, Result};
use rayon::prelude::*;

/// CPU Implementation of Matrix Multiplication.
///
/// This function is the "kernel" that performs the actual computation.
/// It is separated from the `Tensor` struct to allow for easy swapping with
/// optimized libraries (like BLAS) in the future.
///
/// # SOTA Integration Guide
///
/// To integrate a SOTA library like `cblas` or `matrixmultiply`:
/// 1. Replace the body of this function with a call to the library's `sgemm` or `dgemm`.
/// 2. Ensure the memory layout matches (Row-Major vs Column-Major).
///    - `xla-rs` uses Row-Major.
///    - BLAS typically defaults to Column-Major but supports Row-Major via flags.
/// 3. Handle batching:
///    - If the library supports batched matmul, pass the batch count.
///    - Otherwise, loop over the batch dimension here (parallelized with `rayon`).
pub fn cpu_matmul<T, const RANK: usize>(
    lhs_data: &[T],
    rhs_data: &[T],
    lhs_shape: &[usize; RANK],
    rhs_shape: &[usize; RANK],
) -> Result<Vec<T>>
where
    T: KernelElem,
{
    let m = lhs_shape[RANK - 2];
    let k = lhs_shape[RANK - 1];
    let n = rhs_shape[RANK - 1];

    if k != rhs_shape[RANK - 2] {
        return Err(crate::KernelError::ShapeMismatch {
            expected: vec![k],
            got: vec![rhs_shape[RANK - 2]],
        });
    }

    let mut out_shape = *lhs_shape;
    out_shape[RANK - 2] = m;
    out_shape[RANK - 1] = n;
    let size: usize = out_shape.iter().product();
    let mut out_data = vec![T::zero(); size];

    // Optimization: Transpose rhs to allow sequential access (cache friendly)
    // We need to transpose the RHS data. Since we don't have a Tensor object here,
    // we call the kernel directly.
    // rhs is [..., K, N], we want [..., N, K]
    let rhs_t_data = super::cpu_transpose::cpu_transpose(rhs_data, rhs_shape)?;

    // Parallelize over rows of the output matrices across all batches
    // Output shape is [Batch..., M, N]
    // We iterate over (Batch... * M) rows, each of size N

    out_data
        .as_mut_slice()
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(global_row_idx, out_row)| {
            let batch_idx = global_row_idx / m;
            let row_in_matrix = global_row_idx % m;

            // Calculate offsets for input tensors
            let a_batch_offset = batch_idx * m * k;
            // rhs_t has shape [..., N, K], so batch offset is batch_idx * N * K
            let b_t_batch_offset = batch_idx * n * k;

            let a_row_start = a_batch_offset + row_in_matrix * k;
            let a_slice = &lhs_data[a_row_start..a_row_start + k];

            for (col_in_matrix, out_elem) in out_row.iter_mut().enumerate() {
                // We want dot product of:
                // A row: `row_in_matrix`
                // B col: `col_in_matrix` -> which is rhs_t row `col_in_matrix`

                let b_t_row_start = b_t_batch_offset + col_in_matrix * k;
                let b_t_slice = &rhs_t_data[b_t_row_start..b_t_row_start + k];

                let mut sum = T::zero();
                // Vectorizable loop
                for (&val_a, &val_b) in a_slice.iter().zip(b_t_slice.iter()) {
                    sum += val_a * val_b;
                }
                *out_elem = sum;
            }
        });

    Ok(out_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernelError;

    #[test]
    fn test_matmul_simple() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let shape_a = [2, 2];
        let shape_b = [2, 2];

        let result = cpu_matmul(&a, &b, &shape_a, &shape_b).unwrap();
        // Expected:
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_batch() {
        // Batch size 2, 2x2 matrices
        let a = vec![
            1.0, 0.0, 0.0, 1.0, // Identity
            2.0, 0.0, 0.0, 2.0, // Scaled Identity
        ];
        let b = vec![
            1.0, 2.0, 3.0, 4.0, // Matrix B1
            5.0, 6.0, 7.0, 8.0, // Matrix B2
        ];
        let shape_a = [2, 2, 2];
        let shape_b = [2, 2, 2];

        let result = cpu_matmul(&a, &b, &shape_a, &shape_b).unwrap();
        // Expected:
        // Batch 1: I * B1 = B1
        // Batch 2: 2I * B2 = 2 * B2
        let expected = vec![1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_shape_mismatch() {
        let a = vec![1.0; 4]; // 2x2
        let b = vec![1.0; 6]; // 3x2
        let shape_a = [2, 2];
        let shape_b = [3, 2]; // Inner dim mismatch: 2 != 3

        let err = cpu_matmul(&a, &b, &shape_a, &shape_b);
        assert!(matches!(err, Err(KernelError::ShapeMismatch { .. })));
    }
}
