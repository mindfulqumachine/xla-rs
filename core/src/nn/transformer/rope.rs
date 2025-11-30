use crate::tensor::{Cpu, Result, Tensor, TensorElem};
use num_traits::Float;

/// Rotary Positional Embedding (RoPE).
///
/// # What is RoPE?
///
/// RoPE encodes position information by rotating the Query and Key vectors in a high-dimensional space.
///
/// $$ \begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} $$
///
/// Unlike absolute positional embeddings (added to input), RoPE is multiplicative and relative.
/// It allows the model to generalize better to sequence lengths longer than seen during training.
///
/// # Arguments
/// * `x`: Input tensor of shape `[Batch, Heads, SeqLen, HeadDim]`.
/// * `freqs_cos`: Precomputed cosine values `[SeqLen, HeadDim/2]`.
/// * `freqs_sin`: Precomputed sine values `[SeqLen, HeadDim/2]`.
pub fn apply_rope<T: TensorElem>(
    x: &Tensor<T, 4, Cpu>,
    freqs_cos: &Tensor<T, 2, Cpu>, // [SeqLen, HeadDim/2]
    freqs_sin: &Tensor<T, 2, Cpu>,
) -> Result<Tensor<T, 4, Cpu>> {
    // x shape: [Batch, Heads, SeqLen, HeadDim]
    let [b, h, s, d] = *x.shape();

    // This implementation assumes `d` is even.

    let mut out = Tensor::zeros([b, h, s, d]);

    // Parallelize
    use rayon::prelude::*;

    // We iterate over the flattened buffer for efficiency if possible, but indices are tricky.
    // Let's iterate over Batch, Head, SeqLen.

    out.data_mut()
        .par_chunks_mut(s * d) // Chunk by (SeqLen * HeadDim) -> one head's worth of data
        .enumerate()
        .for_each(|(_bh_idx, head_data)| {
            // _bh_idx tracks batch and head, but we don't need them for freq lookup usually,
            // unless freq depends on head (it doesn't usually).

            // head_data is [SeqLen, HeadDim] flat
            for t in 0..s {
                let offset = t * d;
                let freqs_idx = t; // corresponds to position

                for i in 0..(d / 2) {
                    let cos = freqs_cos.data()[freqs_idx * (d / 2) + i];
                    let sin = freqs_sin.data()[freqs_idx * (d / 2) + i];

                    let x0 = head_data[offset + 2 * i];
                    let x1 = head_data[offset + 2 * i + 1];

                    // Rotate
                    head_data[offset + 2 * i] = x0 * cos - x1 * sin;
                    head_data[offset + 2 * i + 1] = x0 * sin + x1 * cos;
                }
            }
        });

    Ok(out)
}

/// Precompute frequency cis for RoPE
pub fn precompute_freqs_cis<T: TensorElem + Float>(
    dim: usize,
    max_seq_len: usize,
    theta: T,
) -> (Tensor<T, 2, Cpu>, Tensor<T, 2, Cpu>) {
    let half_dim = dim / 2;
    let mut cos_out = Tensor::zeros([max_seq_len, half_dim]);
    let mut sin_out = Tensor::zeros([max_seq_len, half_dim]);

    let data_cos = cos_out.data_mut();
    let data_sin = sin_out.data_mut();

    for seq in 0..max_seq_len {
        for i in 0..half_dim {
            // freq = 1.0 / (theta ^ (2*i / dim))
            let freq =
                T::one() / theta.powf(T::from_usize(2 * i).unwrap() / T::from_usize(dim).unwrap());
            let val = T::from_usize(seq).unwrap() * freq;

            data_cos[seq * half_dim + i] = val.cos();
            data_sin[seq * half_dim + i] = val.sin();
        }
    }

    (cos_out, sin_out)
}
