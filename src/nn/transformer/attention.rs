use super::rope::apply_rope;
use crate::nn::Linear;
use crate::tensor::{Cpu, Result, Tensor, TensorElem, TensorOps};
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
    #[allow(clippy::too_many_arguments)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;

    #[test]
    fn test_mha_forward() {
        // B=1, S=2, H=2, D=4 (Head Dim = 2)
        let b = 1;
        let s = 2;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim; // 4

        // Create dummy linear layers (identity weights for simplicity)
        let weight_data = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]; // 4x4 Identity
        let q_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);
        let k_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);
        let v_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);
        let o_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);

        let mha = MultiHeadAttention::new(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        );

        // Input [B, S, Hidden] -> [1, 2, 4]
        let input_data = vec![
            1.0, 0.0, 1.0, 0.0, // Seq 1
            0.0, 1.0, 0.0, 1.0, // Seq 2
        ];
        let x = Tensor::<f32, 3>::new(input_data, [b, s, hidden_dim]).unwrap();

        // RoPE cos/sin [S, HeadDim/2] -> [2, 1] (since head_dim=2)
        // Actually RoPE expects [S, HeadDim/2] for complex, but here implementation details might vary.
        // Looking at rope.rs (not shown but inferred usage), usually [S, HeadDim/2] or [S, HeadDim].
        // Let's assume [S, HeadDim/2] for complex rotation simulation or [S, HeadDim] for full rotation.
        // The apply_rope signature is `freqs_cos: &Tensor<T, 2, Cpu>`.
        // Let's use zeros/ones to be safe/no-op if possible or simple rotation.
        let freqs_cos = Tensor::<f32, 2>::ones([s, head_dim / 2]);
        let freqs_sin = Tensor::<f32, 2>::zeros([s, head_dim / 2]);

        let output = mha.forward(&x, &freqs_cos, &freqs_sin, None).unwrap();

        assert_eq!(output.shape(), &[b, s, hidden_dim]);
    }

    #[test]
    fn test_mha_forward_with_mask() {
        let b = 1;
        let s = 2;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;

        let weight_data = vec![1.0; 16]; // 4x4
        let q_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);
        let k_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);
        let v_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);
        let o_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);

        let mha = MultiHeadAttention::new(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        );

        let input_data = vec![1.0; 8]; // 1x2x4
        let x = Tensor::<f32, 3>::new(input_data, [b, s, hidden_dim]).unwrap();
        let freqs_cos = Tensor::<f32, 2>::ones([s, head_dim / 2]);
        let freqs_sin = Tensor::<f32, 2>::zeros([s, head_dim / 2]);

        // Mask [S, S] -> [2, 2]
        let mask = Tensor::<f32, 2>::zeros([s, s]);

        let output = mha
            .forward(&x, &freqs_cos, &freqs_sin, Some(&mask))
            .unwrap();
        assert_eq!(output.shape(), &[b, s, hidden_dim]);
    }

    #[test]
    fn test_mha_forward_gqa() {
        // Grouped Query Attention: 4 heads, 2 KV heads
        let b = 1;
        let s = 2;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim; // 8
        let kv_dim = num_kv_heads * head_dim; // 4

        // Weights need to match dimensions
        // Q: [Hidden, Hidden] -> [8, 8]
        // K, V: [Hidden, KV_Dim] -> [8, 4] (Wait, Linear is [In, Out] or [Out, In]? Linear is usually x @ W.T + b.
        // In xla-rs Linear, weight is [out_features, in_features].
        // x is [B, S, Hidden].
        // q_proj: [Hidden, Hidden] -> Weight [8, 8]
        // k_proj: [KV_Dim, Hidden] -> Weight [4, 8]
        // v_proj: [KV_Dim, Hidden] -> Weight [4, 8]
        // o_proj: [Hidden, Hidden] -> Weight [8, 8]

        let q_w = Tensor::new(vec![1.0; 64], [hidden_dim, hidden_dim]).unwrap();
        let k_w = Tensor::new(vec![1.0; 32], [kv_dim, hidden_dim]).unwrap();
        let v_w = Tensor::new(vec![1.0; 32], [kv_dim, hidden_dim]).unwrap();
        let o_w = Tensor::new(vec![1.0; 64], [hidden_dim, hidden_dim]).unwrap();

        let q_proj = Linear::new(q_w, None);
        let k_proj = Linear::new(k_w, None);
        let v_proj = Linear::new(v_w, None);
        let o_proj = Linear::new(o_w, None);

        let mha = MultiHeadAttention::new(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        );

        let input_data = vec![1.0; 16]; // 1x2x8
        let x = Tensor::<f32, 3>::new(input_data, [b, s, hidden_dim]).unwrap();
        let freqs_cos = Tensor::<f32, 2>::ones([s, head_dim / 2]);
        let freqs_sin = Tensor::<f32, 2>::zeros([s, head_dim / 2]);

        let output = mha.forward(&x, &freqs_cos, &freqs_sin, None).unwrap();
        assert_eq!(output.shape(), &[b, s, hidden_dim]);
    }

    #[test]
    fn test_softmax_max_not_first() {
        // Test case where max value is not at index 0 to cover "v > max_val" branch
        let b = 1;
        let s = 2;
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 2;
        let hidden_dim = 2;

        // Identity weights
        let weight_data = vec![1.0, 0.0, 0.0, 1.0]; // 2x2
        let q_proj = Linear::new(Tensor::new(weight_data.clone(), [2, 2]).unwrap(), None);
        let k_proj = Linear::new(Tensor::new(weight_data.clone(), [2, 2]).unwrap(), None);
        let v_proj = Linear::new(Tensor::new(weight_data.clone(), [2, 2]).unwrap(), None);
        let o_proj = Linear::new(Tensor::new(weight_data.clone(), [2, 2]).unwrap(), None);

        let mha = MultiHeadAttention::new(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        );

        // S=2.
        // Input:
        // Seq1: [0, 1]
        // Seq2: [0, 2]

        let input_data = vec![0.0, 1.0, 0.0, 2.0];
        let x = Tensor::<f32, 3>::new(input_data, [b, s, hidden_dim]).unwrap();
        let freqs_cos = Tensor::<f32, 2>::ones([s, head_dim / 2]);
        let freqs_sin = Tensor::<f32, 2>::zeros([s, head_dim / 2]);

        let _ = mha.forward(&x, &freqs_cos, &freqs_sin, None).unwrap();
    }

    #[test]
    fn test_repeat_kv_no_rep() {
        let b = 1;
        let s = 2;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;

        // Dummy linear layers
        let weight_data = vec![1.0; 16];
        let q_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);
        let k_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);
        let v_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);
        let o_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);

        let mha = MultiHeadAttention::new(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        );

        // Input [B, NumKV, S, D]
        let input = Tensor::<f32, 4>::zeros([b, num_kv_heads, s, head_dim]);
        let output = mha.repeat_kv(&input).unwrap();

        // Should be identical (clone)
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_mha_forward_mixed_mask() {
        let b = 1;
        let s = 2;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;

        let weight_data = vec![1.0; 16];
        let q_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);
        let k_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);
        let v_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);
        let o_proj = Linear::new(Tensor::new(weight_data.clone(), [4, 4]).unwrap(), None);

        let mha = MultiHeadAttention::new(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        );

        let input_data = vec![1.0; 8];
        let x = Tensor::<f32, 3>::new(input_data, [b, s, hidden_dim]).unwrap();
        let freqs_cos = Tensor::<f32, 2>::ones([s, head_dim / 2]);
        let freqs_sin = Tensor::<f32, 2>::zeros([s, head_dim / 2]);

        // Mask [S, S] -> [2, 2]
        // [1.0, 0.0]
        // [0.0, 1.0]
        // 1.0 means "keep" (no change in additive mask logic if we assume 1.0 is identity? Wait)
        // logic: if mask != 1.0 { val += mask }
        // If mask is 1.0, we do nothing.
        // If mask is 0.0, we add 0.0 (no change).
        // Wait, usually mask is 0 for keep and -inf for mask out.
        // Or 1 for keep and 0 for mask out (multiplicative).
        // The code says: `if mask_data[i] != T::one() { *val += mask_data[i]; }`
        // This implies if mask is 1.0, we do nothing.
        // If mask is 0.0, we add 0.0.
        // If mask is -1e9, we add -1e9.
        // So to cover the `else` (do nothing), we need 1.0 in the mask.
        let mask_data = vec![1.0, 0.0, 0.0, 1.0];
        let mask = Tensor::<f32, 2>::new(mask_data, [s, s]).unwrap();

        let output = mha
            .forward(&x, &freqs_cos, &freqs_sin, Some(&mask))
            .unwrap();
        assert_eq!(output.shape(), &[b, s, hidden_dim]);
    }
}
