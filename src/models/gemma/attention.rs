use super::rope::apply_rope;
use crate::nn::Linear;
use crate::tensor::{Cpu, Result, Tensor, TensorElem};
use num_traits::Float;
use rayon::prelude::*;

#[derive(Debug)]
pub struct MultiHeadAttention<T: TensorElem> {
    pub q_proj: Linear<T>,
    pub k_proj: Linear<T>,
    pub v_proj: Linear<T>,
    pub o_proj: Linear<T>,

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scaling: T,
}

impl<T: TensorElem + Float> MultiHeadAttention<T> {
    pub fn new(
        _dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_proj: Linear<T>,
        k_proj: Linear<T>,
        v_proj: Linear<T>,
        o_proj: Linear<T>,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scaling: T::one() / T::from_usize(head_dim).unwrap().sqrt(),
        }
    }

    pub fn forward(
        &self,
        x: &Tensor<T, 3, Cpu>,
        freqs_cos: &Tensor<T, 2, Cpu>,
        freqs_sin: &Tensor<T, 2, Cpu>,
        mask: Option<&Tensor<T, 2, Cpu>>,
    ) -> Result<Tensor<T, 3, Cpu>> {
        let [b, s, _] = *x.shape();

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [B, S, H, D]
        let q = q.reshape([b, s, self.num_heads, self.head_dim])?;
        let k = k.reshape([b, s, self.num_kv_heads, self.head_dim])?;
        let v = v.reshape([b, s, self.num_kv_heads, self.head_dim])?;

        // Permute to [B, H, S, D] using transpose_axes(1, 2)
        let q = q.transpose_axes(1, 2)?;
        let k = k.transpose_axes(1, 2)?;
        let v = v.transpose_axes(1, 2)?;

        // Apply RoPE (expects [B, H, S, D])
        let q = apply_rope(&q, freqs_cos, freqs_sin)?;
        let k = apply_rope(&k, freqs_cos, freqs_sin)?;

        let (k, v) = if self.num_kv_heads != self.num_heads {
            (self.repeat_kv(&k)?, self.repeat_kv(&v)?)
        } else {
            (k, v)
        };

        // Attention Score: q @ k.T
        // q: [B, H, S, D]
        // k: [B, H, S, D] -> k.transpose() (swaps last two) -> [B, H, D, S]
        let k_t = k.transpose()?;

        let q_flat = q.clone().reshape([b * self.num_heads, s, self.head_dim])?;
        let k_t_flat = k_t.reshape([b * self.num_heads, self.head_dim, s])?;

        let mut scores = q_flat.matmul(&k_t_flat)?;

        scores = scores.map(|val| val * self.scaling);

        if let Some(m) = mask {
            self.apply_mask(&mut scores, m)?;
        }

        self.softmax_inplace(&mut scores)?;

        let v_flat = v.reshape([b * self.num_heads, s, self.head_dim])?;
        let output = scores.matmul(&v_flat)?;

        // output: [B*H, S, D] -> [B, H, S, D]
        let output = output.reshape([b, self.num_heads, s, self.head_dim])?;

        // We need [B, S, H, D].
        // This is transpose_axes(1, 2) again on [B, H, S, D].
        let output = output.transpose_axes(1, 2)?;

        let output = output.reshape([b, s, self.num_heads * self.head_dim])?;

        self.o_proj.forward(&output)
    }

    fn repeat_kv(&self, x: &Tensor<T, 4, Cpu>) -> Result<Tensor<T, 4, Cpu>> {
        let [b, n_kv, s, d] = *x.shape();
        let n_rep = self.num_heads / n_kv;

        if n_rep == 1 {
            return Ok(x.clone());
        }

        let mut out = Tensor::zeros([b, self.num_heads, s, d]);
        let src = x.data();
        let dst = out.data_mut();

        for batch in 0..b {
            for h in 0..self.num_heads {
                let src_h = h / n_rep;
                let src_offset = (batch * n_kv + src_h) * s * d;
                let dst_offset = (batch * self.num_heads + h) * s * d;

                dst[dst_offset..dst_offset + s * d]
                    .copy_from_slice(&src[src_offset..src_offset + s * d]);
            }
        }
        Ok(out)
    }

    fn softmax_inplace(&self, x: &mut Tensor<T, 3, Cpu>) -> Result<()> {
        let [_, _, s] = *x.shape();
        x.data_mut().par_chunks_mut(s).for_each(|row| {
            let mut max_val = row[0];
            for &v in row.iter() {
                if v > max_val {
                    max_val = v;
                }
            }

            let mut sum_exp = T::zero();
            for v in row.iter_mut() {
                let exp_v = (*v - max_val).to_f32().unwrap().exp();
                let exp_v_t = T::from_f32(exp_v).unwrap();
                *v = exp_v_t;
                sum_exp += exp_v_t;
            }

            let inv_sum = T::one() / sum_exp;
            for v in row.iter_mut() {
                *v *= inv_sum;
            }
        });
        Ok(())
    }

    fn apply_mask(&self, scores: &mut Tensor<T, 3, Cpu>, mask: &Tensor<T, 2, Cpu>) -> Result<()> {
        let [_, s, _] = *scores.shape();
        let [ms1, ms2] = *mask.shape();

        if s != ms1 || s != ms2 {
            return Err(crate::tensor::TensorError::ShapeMismatch {
                expected: vec![s, s],
                got: vec![ms1, ms2],
            });
        }

        let mask_data = mask.data();

        scores
            .data_mut()
            .par_chunks_mut(s * s)
            .for_each(|score_matrix| {
                for (i, val) in score_matrix.iter_mut().enumerate() {
                    if mask_data[i] != T::one() {
                        *val += mask_data[i];
                    }
                }
            });
        Ok(())
    }
}
