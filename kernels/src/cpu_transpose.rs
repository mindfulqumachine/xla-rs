use crate::{KernelElem, Result};
use rayon::prelude::*;

/// CPU Implementation of Transpose.
///
/// Swaps the last two dimensions of the input data.
///
/// # SOTA Integration Guide
///
/// Optimized transpose operations often use tiling (blocking) to improve cache usage.
/// Libraries like `hptt` (High Performance Tensor Transpose) can be used here.
pub fn cpu_transpose<T, const RANK: usize>(data: &[T], shape: &[usize; RANK]) -> Result<Vec<T>>
where
    T: KernelElem,
{
    let m = shape[RANK - 2];
    let n = shape[RANK - 1];

    let mut new_shape = *shape;
    new_shape.swap(RANK - 1, RANK - 2);
    let size: usize = new_shape.iter().product();
    let mut out_data = vec![T::zero(); size];

    // We parallelize over the rows of the OUTPUT tensor.
    // The output tensor has shape [Batch..., N, M].
    // So we view it as `batch_size * N` rows, each of length `M`.
    out_data
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
            for (r, out_elem) in out_row.iter_mut().enumerate() {
                // Input is [M, N]. We want element at (r, col_idx).
                // Index = input_batch_offset + r * N + col_idx
                *out_elem = data[input_batch_offset + r * n + col_idx];
            }
        });

    Ok(out_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_simple() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let shape = [2, 3];

        let result = cpu_transpose(&data, &shape).unwrap();
        // Expected 3x2:
        // [1, 4]
        // [2, 5]
        // [3, 6]
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_batch() {
        // Batch size 2, 2x2 matrices
        let data = vec![
            1.0, 2.0, 3.0, 4.0, // Matrix 1
            5.0, 6.0, 7.0, 8.0, // Matrix 2
        ];
        let shape = [2, 2, 2];

        let result = cpu_transpose(&data, &shape).unwrap();
        // Expected:
        // Batch 1: [1, 3, 2, 4]
        // Batch 2: [5, 7, 6, 8]
        let expected = vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0];
        assert_eq!(result, expected);
    }
}
