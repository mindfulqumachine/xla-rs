use crate::nn::Linear;
use crate::tensor::{Cpu, Result, Tensor, TensorElem};
use num_traits::Float;
use rayon::prelude::*;

const PARALLEL_THRESHOLD: usize = 4096;

/// Key-Value Cache for autoregressive generation.
///
/// Stores the Key and Value tensors for all previous tokens in the sequence.
/// This avoids recomputing them at every step, turning O(N^2) inference into O(N).
#[derive(Debug, Clone)]
pub struct KVCache<T: TensorElem> {
    pub k: Tensor<T, 4, Cpu>, // [Batch, KV_Heads, MaxSeqLen, HeadDim]
    pub v: Tensor<T, 4, Cpu>, // [Batch, KV_Heads, MaxSeqLen, HeadDim]
    pub length: usize,        // Current sequence length stored
}

impl<T: TensorElem> KVCache<T> {
    pub fn new(batch: usize, kv_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        Self {
            k: Tensor::zeros([batch, kv_heads, max_seq_len, head_dim]),
            v: Tensor::zeros([batch, kv_heads, max_seq_len, head_dim]),
            length: 0,
        }
    }

    /// Updates the cache with new keys and values.
    ///
    /// # Arguments
    /// * `new_k` - New keys [Batch, KV_Heads, SeqLen, HeadDim]
    /// * `new_v` - New values [Batch, KV_Heads, SeqLen, HeadDim]
    /// * `start_pos` - The starting position in the sequence to write to.
    pub fn update(
        &mut self,
        new_k: &Tensor<T, 4, Cpu>,
        new_v: &Tensor<T, 4, Cpu>,
        start_pos: usize,
    ) -> Result<()> {
        let [_b, _h, s, d] = *new_k.shape();

        let max_seq = self.k.shape()[2];

        let cache_k_data = self.k.data_mut();
        let cache_v_data = self.v.data_mut();
        let new_k_data = new_k.data();
        let new_v_data = new_v.data();

        // Parallelize over Batch and Heads using chunks
        // Each chunk corresponds to one head's full sequence buffer [MaxSeqLen, HeadDim]
        // Chunk size = MaxSeqLen * HeadDim
        let total_elements = _b * _h * s * d;

        let update_fn = |(i, (k_head_buf, v_head_buf)): (usize, (&mut [T], &mut [T]))| {
            // i is global head index (batch * h + head)
            // We need to find the corresponding source data
            // Source is [Batch, Heads, SeqLen, HeadDim]
            // Source is also contiguous per head?
            // new_k shape: [B, H, S, D]
            // Yes, new_k is also laid out as B -> H -> S -> D
            // So the i-th head in cache corresponds to the i-th head in new_k

            // Source chunk size = S * D
            let src_offset = i * s * d;
            let src_k = &new_k_data[src_offset..src_offset + s * d];
            let src_v = &new_v_data[src_offset..src_offset + s * d];

            // Destination offset within the head buffer
            // We are writing to [start_pos..start_pos+s]
            let dst_offset = start_pos * d;
            let dst_len = s * d;

            k_head_buf[dst_offset..dst_offset + dst_len].copy_from_slice(src_k);
            v_head_buf[dst_offset..dst_offset + dst_len].copy_from_slice(src_v);
        };

        if total_elements >= PARALLEL_THRESHOLD {
            cache_k_data
                .par_chunks_mut(max_seq * d)
                .zip(cache_v_data.par_chunks_mut(max_seq * d))
                .enumerate()
                .for_each(update_fn);
        } else {
            cache_k_data
                .chunks_mut(max_seq * d)
                .zip(cache_v_data.chunks_mut(max_seq * d))
                .enumerate()
                .for_each(update_fn);
        }

        self.length = start_pos + s;
        Ok(())
    }

    /// Returns the valid slice of Keys and Values up to current length.
    /// Note: In a zero-copy framework, this would be a view. Here we might have to clone
    /// if our Tensor struct doesn't support slicing views yet.
    /// For this optimization demo, we will assume we pass the whole cache to the kernel
    /// and the kernel knows `length`.
    pub fn get_view(&self) -> (&Tensor<T, 4, Cpu>, &Tensor<T, 4, Cpu>, usize) {
        (&self.k, &self.v, self.length)
    }
}

pub type ForwardWithWeightsOutput<T> = (Tensor<T, 3, Cpu>, Option<Tensor<T, 4, Cpu>>);
pub type SplitQKVOutput<T> = (Tensor<T, 3, Cpu>, Tensor<T, 3, Cpu>, Tensor<T, 3, Cpu>);

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

    /// Optimized Forward Pass
    ///
    /// 1. Fused RoPE + Transpose
    /// 2. KV Cache Update
    /// 3. Fused Attention (Score + Softmax + Weighted Sum)
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

        let (q, k, v) = self.split_qkv(&qkv, q_dim, kv_dim)?;

        // 3. Fused Reshape + Transpose + RoPE
        // We can fuse RoPE into the split if we are clever, but for now let's keep it separate
        // or fuse it into the next step.
        // Actually, `fused_rope_transpose` takes [B, S, H*D].
        // We can pass `q` and `k` to it.

        let q_rope = self.fused_rope_transpose(&q, freqs_cos, freqs_sin, self.num_heads)?;
        let k_rope = self.fused_rope_transpose(&k, freqs_cos, freqs_sin, self.num_kv_heads)?;

        // For V, we just need transpose [B, S, H, D] -> [B, H, S, D]
        // But `v` is currently [B, S, H*D].
        let v_transposed = self.fused_transpose(&v, self.num_kv_heads)?;

        // 3. KV Cache Management
        let (k_in, v_in, current_len) = if let Some(cache) = kv_cache {
            cache.update(&k_rope, &v_transposed, start_pos)?;
            (&cache.k, &cache.v, cache.length)
        } else {
            (&k_rope, &v_transposed, s)
        };

        // 4. Grouped Query Attention with Caching
        // Pass true to return weights
        let (output, weights) = self.fused_attention(&q_rope, k_in, v_in, current_len, true)?;

        // 5. Output Projection
        let output = self.o_proj.forward(&output)?;
        Ok((output, weights))
    }

    /// Fuses Reshape [B, S, H*D] -> [B, S, H, D], Transpose -> [B, H, S, D], and RoPE.
    fn split_qkv(
        &self,
        qkv: &Tensor<T, 3, Cpu>,
        q_dim: usize,
        kv_dim: usize,
    ) -> Result<SplitQKVOutput<T>> {
        let [b, s, total_dim] = *qkv.shape();
        if total_dim != q_dim + 2 * kv_dim {
            return Err(crate::tensor::TensorError::ShapeMismatch {
                expected: vec![q_dim + 2 * kv_dim],
                got: vec![total_dim],
            });
        }

        let mut q = Tensor::zeros([b, s, q_dim]);
        let mut k = Tensor::zeros([b, s, kv_dim]);
        let mut v = Tensor::zeros([b, s, kv_dim]);

        let qkv_data = qkv.data();
        let q_data = q.data_mut();
        let k_data = k.data_mut();
        let v_data = v.data_mut();

        // Parallelize over tokens (B * S)
        // Parallelize over tokens (B * S)
        let total_elements = b * s * total_dim;
        #[allow(clippy::type_complexity)]
        let split_fn = |(i, ((q_row, k_row), v_row)): (usize, ((&mut [T], &mut [T]), &mut [T]))| {
            let src_offset = i * total_dim;
            let src_row = &qkv_data[src_offset..src_offset + total_dim];

            q_row.copy_from_slice(&src_row[0..q_dim]);
            k_row.copy_from_slice(&src_row[q_dim..q_dim + kv_dim]);
            v_row.copy_from_slice(&src_row[q_dim + kv_dim..]);
        };

        if total_elements >= PARALLEL_THRESHOLD {
            q_data
                .par_chunks_mut(q_dim)
                .zip(k_data.par_chunks_mut(kv_dim))
                .zip(v_data.par_chunks_mut(kv_dim))
                .enumerate()
                .for_each(split_fn);
        } else {
            q_data
                .chunks_mut(q_dim)
                .zip(k_data.chunks_mut(kv_dim))
                .zip(v_data.chunks_mut(kv_dim))
                .enumerate()
                .for_each(split_fn);
        }

        Ok((q, k, v))
    }

    fn fused_rope_transpose(
        &self,
        x: &Tensor<T, 3, Cpu>, // [B, S, H*D]
        freqs_cos: &Tensor<T, 2, Cpu>,
        freqs_sin: &Tensor<T, 2, Cpu>,
        num_heads: usize,
    ) -> Result<Tensor<T, 4, Cpu>> {
        let [b, s, _] = *x.shape();
        let d = self.head_dim;

        let mut out = Tensor::zeros([b, num_heads, s, d]);

        let x_data = x.data();
        let out_data = out.data_mut();
        let cos_data = freqs_cos.data();
        let sin_data = freqs_sin.data();

        // Parallelize over Batch and Heads
        // Parallelize over Batch and Heads
        let total_elements = b * num_heads * s * d;
        let rope_fn = |(i, out_head): (usize, &mut [T])| {
            let batch_idx = i / num_heads;
            let head_idx = i % num_heads;

            for t in 0..s {
                // Input index: [batch, t, head * d]
                // But input is flat [B, S, H*D]
                // offset = batch * (S * H * D) + t * (H * D) + head * D
                let in_offset =
                    (batch_idx * s * num_heads * d) + (t * num_heads * d) + (head_idx * d);

                // RoPE index: corresponds to position `t` (relative to start of this seq chunk)
                // In real generation, we'd need absolute position `start_pos + t`.
                // For simplicity here assuming start_pos=0 or handled by caller passing correct freqs slice.
                let freq_idx = t;

                for j in 0..d / 2 {
                    let x0 = x_data[in_offset + 2 * j];
                    let x1 = x_data[in_offset + 2 * j + 1];

                    let cos = cos_data[freq_idx * (d / 2) + j];
                    let sin = sin_data[freq_idx * (d / 2) + j];

                    // Apply RoPE and write to output [t, 2*j] (relative to head chunk)
                    // Output chunk is [S, D], so index is t * d + 2*j
                    let out_idx = t * d + 2 * j;

                    out_head[out_idx] = x0 * cos - x1 * sin;
                    out_head[out_idx + 1] = x0 * sin + x1 * cos;
                }
            }
        };

        if total_elements >= PARALLEL_THRESHOLD {
            out_data.par_chunks_mut(s * d).enumerate().for_each(rope_fn);
        } else {
            out_data.chunks_mut(s * d).enumerate().for_each(rope_fn);
        }

        Ok(out)
    }

    fn fused_transpose(
        &self,
        x: &Tensor<T, 3, Cpu>,
        num_heads: usize,
    ) -> Result<Tensor<T, 4, Cpu>> {
        let [b, s, _] = *x.shape();
        let d = self.head_dim;
        let mut out = Tensor::zeros([b, num_heads, s, d]);

        let x_data = x.data();
        let out_data = out.data_mut();

        let total_elements = b * num_heads * s * d;
        let transpose_fn = |(i, out_head): (usize, &mut [T])| {
            let batch_idx = i / num_heads;
            let head_idx = i % num_heads;

            for t in 0..s {
                let in_offset =
                    (batch_idx * s * num_heads * d) + (t * num_heads * d) + (head_idx * d);
                let out_idx = t * d;

                // Copy head_dim elements
                out_head[out_idx..out_idx + d].copy_from_slice(&x_data[in_offset..in_offset + d]);
            }
        };

        if total_elements >= PARALLEL_THRESHOLD {
            out_data
                .par_chunks_mut(s * d)
                .enumerate()
                .for_each(transpose_fn);
        } else {
            out_data
                .chunks_mut(s * d)
                .enumerate()
                .for_each(transpose_fn);
        }

        Ok(out)
    }

    /// Fused Attention Kernel
    /// Handles GQA (repeating KV) implicitly via indexing, avoiding memory expansion.
    fn fused_attention(
        &self,
        q: &Tensor<T, 4, Cpu>, // [B, H_q, S_q, D]
        k: &Tensor<T, 4, Cpu>, // [B, H_kv, Total_S, D]
        v: &Tensor<T, 4, Cpu>, // [B, H_kv, Total_S, D]
        kv_len: usize,
        return_weights: bool,
    ) -> Result<ForwardWithWeightsOutput<T>> {
        let [b, h_q, s_q, d] = *q.shape();
        let [_, h_kv, _, _] = *k.shape();

        // Output: [B, S_q, H_q * D] (Ready for linear projection)
        let mut output = Tensor::zeros([b, s_q, h_q * d]);

        // Optional weights: [B, H_q, S_q, Total_S]
        // We need to use a Mutex or similar if we want to write to it in parallel,
        // or just construct it. Since we are inside rayon, let's use a pre-allocated tensor if needed.
        // However, Tensor doesn't expose safe parallel mutable access easily without unsafe or splitting.
        // For simplicity in this "Optimized" demo which is still CPU, let's just collect weights if requested.
        // But wait, `output` is being written to in parallel.
        // If we want to write weights in parallel, we need a similar structure.

        let mut weights_tensor = if return_weights {
            Some(Tensor::zeros([b, h_q, s_q, kv_len]))
        } else {
            None
        };

        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();
        let out_data = output.data_mut();

        // We need to handle weights_data carefully for parallel access.
        // We can use `split_at_mut` or similar, but `Tensor` might not expose it easily for arbitrary shapes.
        // Let's assume for now we can get a raw slice or we use `par_chunks_mut` on it too.
        // The weights tensor is [B, H_q, S_q, KV_Len].
        // Flattened: B * H_q * S_q * KV_Len.
        // We iterate over tokens: B * S_q.
        // Inside token, we iterate over H_q.
        // So we can chunk weights by `H_q * KV_Len`? No.
        // Weights layout: B -> H_q -> S_q -> KV_Len.
        // Output layout: B -> S_q -> H_q*D.
        // The loops are:
        // for i in 0..B*S_q: (token index)
        //   batch = i / S_q
        //   seq = i % S_q
        //   for h in 0..H_q:
        //     ...

        // The output iteration order (B, S_q) is NOT contiguous for Weights (B, H_q, S_q).
        // Weights are grouped by Head first.
        // Output is grouped by Sequence first (standard Transformer output).
        // This mismatch makes parallel writing to both hard without unsafe or reordering.

        // Option A: Change loop order to match Weights?
        // If we loop B -> H_q -> S_q, then Output is non-contiguous (scattered writes).
        // Option B: Use unsafe to write to weights.
        // Option C: Don't parallelize weights writing (slow).
        // Option D: Just use a `Vec` of weights and assemble later?

        // Let's stick to the current loop structure (B, S_q) which is good for Output.
        // And we will use `unsafe` to get a mutable pointer to weights data to write to it,
        // knowing that each thread writes to a disjoint part.
        // OR, since this is a demo/educational code, maybe we just don't parallelize if weights are requested?
        // Or we just accept the complexity.

        // Let's try to be safe. We can collect results in the parallel iterator.
        // The iterator produces `(out_token_data, weights_token_data)`.
        // But `out_data` is already a slice.

        // Let's use `unsafe` for the weights pointer, it's the standard way to do "scatter" writes in Rust parallel iterators if we can guarantee disjointness.
        // Each (batch, seq, head) tuple is unique.
        // Weights index: (batch * H_q * S_q * KV_Len) + (head * S_q * KV_Len) + (seq * KV_Len).
        // This is unique for each (batch, seq, head).

        let weights_ptr: *mut T = if let Some(ref mut w) = weights_tensor {
            w.data_mut().as_mut_ptr()
        } else {
            std::ptr::null_mut()
        };

        // Cast to usize to pass through rayon closure (usize is Send + Sync)
        let weights_ptr_addr = weights_ptr as usize;

        let n_rep = h_q / h_kv;
        let scale = self.scaling;

        // Iterate over tokens (Batch * Seq)
        // Iterate over tokens (Batch * Seq)
        let total_elements = b * s_q * h_q * d;
        let attn_fn = |(i, out_token): (usize, &mut [T])| {
            let batch_idx = i / s_q;
            let seq_idx = i % s_q;

            let w_ptr = weights_ptr_addr as *mut T;

            // For this token, we compute all heads
            for head_idx in 0..h_q {
                let kv_head_idx = head_idx / n_rep;

                // Q vector: [B, H, S, D]
                let q_offset = (batch_idx * h_q * s_q * d) + (head_idx * s_q * d) + (seq_idx * d);
                let q_vec = &q_data[q_offset..q_offset + d];

                // K/V vectors: [B, H_kv, Total_S, D]
                let k_offset_base =
                    (batch_idx * h_kv * k.shape()[2] * d) + (kv_head_idx * k.shape()[2] * d);
                let v_offset_base =
                    (batch_idx * h_kv * v.shape()[2] * d) + (kv_head_idx * v.shape()[2] * d);

                // 1. Compute Scores
                let mut scores = vec![T::zero(); kv_len];
                let mut max_score = T::min_value();

                for pos in 0..kv_len {
                    let k_vec = &k_data[k_offset_base + pos * d..k_offset_base + (pos + 1) * d];
                    let mut score = T::zero();
                    for j in 0..d {
                        score += q_vec[j] * k_vec[j];
                    }
                    score *= scale;
                    scores[pos] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                // 2. Softmax
                let mut sum_exp = T::zero();
                for score in scores.iter_mut() {
                    let exp_val = (*score - max_score).to_f32().unwrap().exp();
                    let exp_t = T::from_f32(exp_val).unwrap();
                    *score = exp_t;
                    sum_exp += exp_t;
                }
                let inv_sum = T::one() / sum_exp;

                // Normalize scores to get probabilities (attention weights)
                for score in scores.iter_mut() {
                    *score *= inv_sum;
                }

                // 3. Weighted Sum
                let mut out_vec = vec![T::zero(); d];
                for pos in 0..kv_len {
                    let weight = scores[pos];
                    let v_vec = &v_data[v_offset_base + pos * d..v_offset_base + (pos + 1) * d];
                    for j in 0..d {
                        out_vec[j] += weight * v_vec[j];
                    }
                }

                // Save weights if requested
                if !w_ptr.is_null() {
                    // Index: (batch * H_q * S_q * KV_Len) + (head * S_q * KV_Len) + (seq * KV_Len)
                    // Note: S_q is the sequence length of the query (x), which is `s` in forward.
                    // KV_Len is the total length of K/V.
                    let w_offset = (batch_idx * h_q * s_q * kv_len)
                        + (head_idx * s_q * kv_len)
                        + (seq_idx * kv_len);

                    unsafe {
                        let dst = std::slice::from_raw_parts_mut(w_ptr.add(w_offset), kv_len);
                        dst.copy_from_slice(&scores);
                    }
                }

                // Write to output buffer
                let out_head_offset = head_idx * d;
                out_token[out_head_offset..out_head_offset + d].copy_from_slice(&out_vec[..d]);
            }
        };

        if total_elements >= PARALLEL_THRESHOLD {
            out_data
                .par_chunks_mut(h_q * d)
                .enumerate()
                .for_each(attn_fn);
        } else {
            out_data.chunks_mut(h_q * d).enumerate().for_each(attn_fn);
        }

        Ok((output, weights_tensor))
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

        // Mock attention struct just to access fused_rope_transpose
        // We can't easily instantiate it without valid linears, but we can make dummy ones
        // Model dim is 2.
        let make_dummy = || Linear::new(Tensor::zeros([2, 2]), None);
        let attn = OptimizedMultiHeadAttention::<f32>::new(
            num_heads,
            num_heads,
            head_dim,
            make_dummy(),
            make_dummy(),
            make_dummy(),
            make_dummy(),
        )?;

        let out = attn.fused_rope_transpose(&x, &freqs_cos, &freqs_sin, num_heads)?;

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
