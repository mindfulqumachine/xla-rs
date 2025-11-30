pub use crate::kernels::attention::KVCache;
use crate::kernels::attention::{
    ForwardWithWeightsOutput, fused_attention, fused_rope_transpose, fused_transpose, split_qkv,
};
use crate::nn::Linear;
use crate::tensor::{Cpu, Result, Tensor, TensorElem};
use num_traits::Float;

/// Optimized Multi-Head Attention.
///
/// # Optimizations
///
/// This implementation includes several optimizations over the standard `MultiHeadAttention`:
/// 1. **Fused Projections**: Q, K, and V projections are fused into a single linear layer.
/// 2. **KV Caching**: Supports efficient incremental decoding by caching Key/Value states.
/// 3. **Fused Kernels**: Uses specialized kernels for RoPE, Transpose, and Attention to reduce memory overhead.
///
/// # Why Fused Projections?
///
/// Instead of launching 3 separate matrix multiplications (small kernels), we launch 1 large matrix multiplication.
/// This improves GPU/CPU utilization and reduces kernel launch overhead.
#[derive(Debug)]
pub struct OptimizedMultiHeadAttention<T: TensorElem> {
    pub qkv_proj: Linear<T>,
    pub o_proj: Linear<T>,

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scaling: T,
}

impl<T: TensorElem + Float> OptimizedMultiHeadAttention<T> {
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_proj: Linear<T>,
        k_proj: Linear<T>,
        v_proj: Linear<T>,
        o_proj: Linear<T>,
    ) -> Result<Self> {
        // Fuse Q, K, V projections into one
        let q_w = q_proj.weight.data();
        let k_w = k_proj.weight.data();
        let v_w = v_proj.weight.data();

        let mut qkv_w_data = Vec::with_capacity(q_w.len() + k_w.len() + v_w.len());
        qkv_w_data.extend_from_slice(q_w);
        qkv_w_data.extend_from_slice(k_w);
        qkv_w_data.extend_from_slice(v_w);

        let in_features = q_proj.weight.shape()[1];
        // Wait, GQA means K and V might have fewer heads!
        // q_proj: [H * D, I]
        // k_proj: [KV * D, I]
        // v_proj: [KV * D, I]

        let q_out = q_proj.weight.shape()[0];
        let k_out = k_proj.weight.shape()[0];
        let v_out = v_proj.weight.shape()[0];

        let qkv_out = q_out + k_out + v_out;
        let qkv_weight = Tensor::new(qkv_w_data, [qkv_out, in_features])?;

        let qkv_bias =
            if let (Some(qb), Some(kb), Some(vb)) = (&q_proj.bias, &k_proj.bias, &v_proj.bias) {
                let q_b = qb.data();
                let k_b = kb.data();
                let v_b = vb.data();
                let mut qkv_b_data = Vec::with_capacity(q_b.len() + k_b.len() + v_b.len());
                qkv_b_data.extend_from_slice(q_b);
                qkv_b_data.extend_from_slice(k_b);
                qkv_b_data.extend_from_slice(v_b);
                Some(Tensor::new(qkv_b_data, [qkv_out])?)
            } else {
                None
            };

        let qkv_proj = Linear::new(qkv_weight, qkv_bias);

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scaling: T::one() / T::from_usize(head_dim).unwrap().sqrt(),
        })
    }

    /// Optimized Forward Pass.
    ///
    /// # Workflow
    /// 1. **Fused QKV**: Compute Q, K, V in one go.
    /// 2. **Split**: Separate Q, K, V.
    /// 3. **Fused RoPE**: Apply Rotary Embeddings and Transpose simultaneously.
    /// 4. **KV Cache**: Update and retrieve cached keys/values.
    /// 5. **Fused Attention**: Compute attention scores and output.
    pub fn forward(
        &self,
        x: &Tensor<T, 3, Cpu>,
        freqs_cos: &Tensor<T, 2, Cpu>,
        freqs_sin: &Tensor<T, 2, Cpu>,
        kv_cache: &mut Option<KVCache<T>>,
        start_pos: usize,
    ) -> Result<Tensor<T, 3, Cpu>> {
        let (output, _) =
            self.forward_with_weights(x, freqs_cos, freqs_sin, kv_cache, start_pos)?;
        Ok(output)
    }

    pub fn forward_with_weights(
        &self,
        x: &Tensor<T, 3, Cpu>,
        freqs_cos: &Tensor<T, 2, Cpu>,
        freqs_sin: &Tensor<T, 2, Cpu>,
        kv_cache: &mut Option<KVCache<T>>,
        start_pos: usize,
    ) -> Result<ForwardWithWeightsOutput<T>> {
        let [_b, s, _] = *x.shape();

        // 1. Fused QKV Projection
        let qkv = self.qkv_proj.forward(x)?; // [B, S, Q_dim + K_dim + V_dim]

        // 2. Split Q, K, V
        // We need to split the last dimension.
        // Q: [B, S, num_heads * head_dim]
        // K: [B, S, num_kv_heads * head_dim]
        // V: [B, S, num_kv_heads * head_dim]

        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let (q, k, v) = split_qkv(&qkv, q_dim, kv_dim)?;

        // 3. Fused Reshape + Transpose + RoPE
        // We can fuse RoPE into the split if we are clever, but for now let's keep it separate
        // or fuse it into the next step.
        // Actually, `fused_rope_transpose` takes [B, S, H*D].
        // We can pass `q` and `k` to it.

        let q_rope = fused_rope_transpose(&q, freqs_cos, freqs_sin, self.num_heads, self.head_dim)?;
        let k_rope =
            fused_rope_transpose(&k, freqs_cos, freqs_sin, self.num_kv_heads, self.head_dim)?;

        // For V, we just need transpose [B, S, H, D] -> [B, H, S, D]
        // But `v` is currently [B, S, H*D].
        let v_transposed = fused_transpose(&v, self.num_kv_heads, self.head_dim)?;

        // 3. KV Cache Management
        let (k_in, v_in, current_len) = if let Some(cache) = kv_cache {
            cache.update(&k_rope, &v_transposed, start_pos)?;
            (&cache.k, &cache.v, cache.length)
        } else {
            (&k_rope, &v_transposed, s)
        };

        // 4. Grouped Query Attention with Caching
        // Pass true to return weights
        let (output, weights) =
            fused_attention(&q_rope, k_in, v_in, current_len, true, self.scaling)?;

        // 5. Output Projection
        let output = self.o_proj.forward(&output)?;
        Ok((output, weights))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    fn tensor_full<const R: usize>(shape: [usize; R], val: f32) -> Tensor<f32, R> {
        let size = shape.iter().product();
        Tensor::new(vec![val; size], shape).unwrap()
    }

    fn tensor_randn<const R: usize>(shape: [usize; R]) -> Tensor<f32, R> {
        let size = shape.iter().product();
        // Use deterministic values for tests
        let data = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();
        Tensor::new(data, shape).unwrap()
    }

    fn tensor_eye(dim: usize) -> Tensor<f32, 2> {
        let mut data = vec![0.0; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Tensor::new(data, [dim, dim]).unwrap()
    }

    #[test]
    fn test_kv_cache_update() -> Result<()> {
        let batch = 1;
        let kv_heads = 2;
        let max_seq_len = 10;
        let head_dim = 4;

        let mut cache = KVCache::<f32>::new(batch, kv_heads, max_seq_len, head_dim);

        // Update with 2 tokens
        let new_k = Tensor::ones([batch, kv_heads, 2, head_dim]);
        let new_v = tensor_full([batch, kv_heads, 2, head_dim], 2.0);

        cache.update(&new_k, &new_v, 0)?;

        assert_eq!(cache.length, 2);

        let (k_view, v_view, len) = cache.get_view();
        assert_eq!(len, 2);

        // Check content at pos 0 and 1
        let k_data = k_view.data();
        let v_data = v_view.data();

        // Check first token (pos 0)
        // Index: batch*H*S*D + head*S*D + pos*D
        // For batch=0, head=0, pos=0
        assert_eq!(k_data[0], 1.0);
        assert_eq!(v_data[0], 2.0);

        // Check second token (pos 1)
        // Index: head=0, pos=1 => offset 1*4 = 4
        assert_eq!(k_data[4], 1.0);
        assert_eq!(v_data[4], 2.0);

        // Update again at pos 2
        let new_k2 = tensor_full([batch, kv_heads, 1, head_dim], 3.0);
        let new_v2 = tensor_full([batch, kv_heads, 1, head_dim], 4.0);
        cache.update(&new_k2, &new_v2, 2)?;

        assert_eq!(cache.length, 3);
        let k_data = cache.k.data();
        // Check third token (pos 2)
        // Index: head=0, pos=2 => offset 2*4 = 8
        assert_eq!(k_data[8], 3.0);

        Ok(())
    }

    #[test]
    fn test_optimized_attention_shapes() -> Result<()> {
        let batch = 1;
        let seq_len = 3;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = 8;
        let model_dim = num_heads * head_dim;

        let q_proj = Linear::new(
            tensor_randn([model_dim, model_dim]),
            Some(Tensor::zeros([model_dim])),
        );
        let k_proj = Linear::new(
            tensor_randn([model_dim, model_dim]),
            Some(Tensor::zeros([model_dim])),
        );
        let v_proj = Linear::new(
            tensor_randn([model_dim, model_dim]),
            Some(Tensor::zeros([model_dim])),
        );
        let o_proj = Linear::new(
            tensor_randn([model_dim, model_dim]),
            Some(Tensor::zeros([model_dim])),
        );

        let attn = OptimizedMultiHeadAttention::new(
            num_heads,
            num_kv_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        )?;

        let x = tensor_randn([batch, seq_len, model_dim]);
        // Freqs for RoPE: [SeqLen, HeadDim/2]
        let freqs_cos = Tensor::ones([seq_len, head_dim / 2]);
        let freqs_sin = Tensor::zeros([seq_len, head_dim / 2]);

        let mut kv_cache = None;

        let output = attn.forward(&x, &freqs_cos, &freqs_sin, &mut kv_cache, 0)?;

        assert_eq!(output.shape(), &[batch, seq_len, model_dim]);

        Ok(())
    }

    #[test]
    fn test_gqa_shapes() -> Result<()> {
        let batch = 1;
        let seq_len = 3;
        let num_heads = 4;
        let num_kv_heads = 2; // GQA: 2 query heads per KV head
        let head_dim = 8;
        let model_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q_proj = Linear::new(
            tensor_randn([model_dim, model_dim]),
            Some(Tensor::zeros([model_dim])),
        );
        let k_proj = Linear::new(
            tensor_randn([kv_dim, model_dim]),
            Some(Tensor::zeros([kv_dim])),
        );
        let v_proj = Linear::new(
            tensor_randn([kv_dim, model_dim]),
            Some(Tensor::zeros([kv_dim])),
        );
        let o_proj = Linear::new(
            tensor_randn([model_dim, model_dim]),
            Some(Tensor::zeros([model_dim])),
        );

        let attn = OptimizedMultiHeadAttention::new(
            num_heads,
            num_kv_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        )?;

        let x = tensor_randn([batch, seq_len, model_dim]);
        let freqs_cos = Tensor::ones([seq_len, head_dim / 2]);
        let freqs_sin = Tensor::zeros([seq_len, head_dim / 2]);

        let mut kv_cache = None;

        let output = attn.forward(&x, &freqs_cos, &freqs_sin, &mut kv_cache, 0)?;

        assert_eq!(output.shape(), &[batch, seq_len, model_dim]);
        Ok(())
    }

    #[test]
    fn test_rope_rotation() -> Result<()> {
        // Test that RoPE actually modifies the input when sin != 0
        let batch = 1;
        let seq_len = 1;
        let num_heads = 1;
        let head_dim = 2;

        // Input [1, 1, 2] -> [[1.0, 0.0]]
        let x = Tensor::new(vec![1.0, 0.0], [batch, seq_len, num_heads * head_dim]).unwrap();

        // Rotate by 90 degrees: cos=0, sin=1
        let freqs_cos = Tensor::zeros([seq_len, head_dim / 2]);
        let freqs_sin = Tensor::ones([seq_len, head_dim / 2]);

        // We can test the kernel directly now!
        let out = fused_rope_transpose(&x, &freqs_cos, &freqs_sin, num_heads, head_dim)?;

        // Expected rotation:
        // x0' = x0*cos - x1*sin = 1*0 - 0*1 = 0
        // x1' = x0*sin + x1*cos = 1*1 + 0*0 = 1
        // Output should be [[0.0, 1.0]] (transposed to [B, H, S, D] -> [1, 1, 1, 2])

        let out_data = out.data();
        assert!((out_data[0] - 0.0).abs() < 1e-6);
        assert!((out_data[1] - 1.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_attention_weights() -> Result<()> {
        let batch = 1;
        let seq_len = 2;
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 4;
        let model_dim = num_heads * head_dim;

        // Identity projections to make reasoning easier
        let eye = tensor_eye(model_dim);
        let zeros = Tensor::zeros([model_dim]);
        let q_proj = Linear::new(eye.clone(), Some(zeros.clone()));
        let k_proj = Linear::new(eye.clone(), Some(zeros.clone()));
        let v_proj = Linear::new(eye.clone(), Some(zeros.clone()));
        let o_proj = Linear::new(eye.clone(), Some(zeros.clone()));

        let attn = OptimizedMultiHeadAttention::new(
            num_heads,
            num_kv_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        )?;

        let x = tensor_randn([batch, seq_len, model_dim]);
        let freqs_cos = Tensor::ones([seq_len, head_dim / 2]);
        let freqs_sin = Tensor::zeros([seq_len, head_dim / 2]);
        let mut kv_cache = None;

        let (_, weights) =
            attn.forward_with_weights(&x, &freqs_cos, &freqs_sin, &mut kv_cache, 0)?;

        assert!(weights.is_some());
        let w = weights.unwrap();
        // Weights shape: [B, H, S, Total_S] -> [1, 1, 2, 2]
        assert_eq!(w.shape(), &[batch, num_heads, seq_len, seq_len]);

        let w_data = w.data();
        // Check softmax sum = 1 for each query position
        // Row 0: w[0,0], w[0,1]
        let sum0 = w_data[0] + w_data[1];
        assert!((sum0 - 1.0).abs() < 1e-5);

        // Row 1: w[1,0], w[1,1]
        let sum1 = w_data[2] + w_data[3];
        assert!((sum1 - 1.0).abs() < 1e-5);

        Ok(())
    }
}
