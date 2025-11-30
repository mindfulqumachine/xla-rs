use crate::models::traits::CausalLM;
pub mod tokenizer;
use crate::nn::transformer::attention::MultiHeadAttention;
use crate::nn::{Activation, Embedding, Linear, RMSNorm};
use crate::tensor::{Cpu, Result, Tensor, TensorElem};
use memmap2::Mmap;
use num_traits::Float;
use safetensors::SafeTensors;
use std::fs::File;
use std::ops::Add;

#[derive(Debug, Clone)]
pub struct GemmaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
}

impl GemmaConfig {
    pub fn gemma_70b() -> Self {
        Self {
            hidden_size: 8192,
            intermediate_size: 32768,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            vocab_size: 256000,
        }
    }

    pub fn gemma_2b() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 16384,
            num_hidden_layers: 18,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            vocab_size: 256000,
        }
    }

    pub fn tiny_test() -> Self {
        Self {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rms_norm_eps: 1e-6,
            vocab_size: 100,
        }
    }
}

#[derive(Debug)]
pub struct MLP<T: TensorElem> {
    pub gate_proj: Linear<T>,
    pub up_proj: Linear<T>,
    pub down_proj: Linear<T>,
}

impl<T: TensorElem + Float> MLP<T> {
    pub fn forward(&self, x: &Tensor<T, 3, Cpu>) -> Result<Tensor<T, 3, Cpu>> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        let gate = Activation::silu(&gate);
        let fused = (&gate * &up)?;

        self.down_proj.forward(&fused)
    }
}

#[derive(Debug)]
pub struct GemmaBlock<T: TensorElem> {
    pub self_attn: MultiHeadAttention<T>,
    pub mlp: MLP<T>,
    pub input_layernorm: RMSNorm<T>,
    pub post_attention_layernorm: RMSNorm<T>,
}

impl<T: TensorElem + Float> GemmaBlock<T> {
    pub fn forward(
        &self,
        x: &Tensor<T, 3, Cpu>,
        freqs_cos: &Tensor<T, 2, Cpu>,
        freqs_sin: &Tensor<T, 2, Cpu>,
        mask: Option<&Tensor<T, 2, Cpu>>,
    ) -> Result<Tensor<T, 3, Cpu>> {
        let residual = x;

        let norm_x = self.input_layernorm.forward(x)?;
        let attn_out = self
            .self_attn
            .forward(&norm_x, Some(freqs_cos), Some(freqs_sin), mask)?;

        let x = (residual.add(&attn_out))?;

        let residual = &x;
        let norm_x = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&norm_x)?;

        let x = (residual.add(&mlp_out))?;

        Ok(x)
    }
}

/// The full Gemma Model.
///
/// Consists of a stack of `GemmaBlock` layers followed by a final RMSNorm.
/// Note: This struct represents the transformer body. The embedding layer and language model head
/// are typically handled separately or wrapped in a `GemmaForCausalLM` struct (not yet implemented).
#[derive(Debug)]
pub struct GemmaModel<T: TensorElem> {
    pub layers: Vec<GemmaBlock<T>>,
    pub norm: RMSNorm<T>,
}

impl<T: TensorElem + Float> GemmaModel<T> {
    pub fn forward(
        &self,
        x: &Tensor<T, 3, Cpu>,
        freqs_cos: &Tensor<T, 2, Cpu>,
        freqs_sin: &Tensor<T, 2, Cpu>,
        mask: Option<&Tensor<T, 2, Cpu>>,
    ) -> Result<Tensor<T, 3, Cpu>> {
        let mut hidden = x.clone();

        for layer in &self.layers {
            hidden = layer.forward(&hidden, freqs_cos, freqs_sin, mask)?;
        }

        self.norm.forward(&hidden)
    }
}

#[derive(Debug)]
/// A causal language model based on the Gemma architecture.
///
/// # Implementation Note
/// This is a simplified implementation for educational purposes. It differs from the official
/// Gemma 2 implementation in the following ways:
/// - Uses standard Global Attention (no interleaved local sliding window attention).
/// - Text-only (no SigLIP vision encoder support).
/// - Simplified RoPE implementation.
pub struct GemmaForCausalLM<T: TensorElem> {
    pub model: GemmaModel<T>,
    pub lm_head: Linear<T>,
    pub embed_tokens: Embedding<T>,
    pub config: GemmaConfig,
}

impl<T: TensorElem + Float> CausalLM<T> for GemmaForCausalLM<T> {
    fn forward(&self, input_ids: &Tensor<usize, 2, Cpu>) -> Result<Tensor<T, 3, Cpu>> {
        let [_, seq_len] = *input_ids.shape();

        // 1. Embed tokens
        let inputs_embeds = self.embed_tokens.forward(input_ids)?;

        // 2. Prepare RoPE frequencies
        // TODO: Implement proper RoPE frequency generation based on config and position
        // For now, using dummy frequencies to satisfy the signature
        let head_dim = self.config.head_dim;
        let freqs_cos = Tensor::ones([seq_len, head_dim / 2]);
        let freqs_sin = Tensor::zeros([seq_len, head_dim / 2]);

        // 3. Forward through model body
        let hidden_states = self
            .model
            .forward(&inputs_embeds, &freqs_cos, &freqs_sin, None)?;

        // 4. LM Head
        self.lm_head.forward(&hidden_states)
    }

    fn generate(
        &self,
        prompt_ids: &Tensor<usize, 2, Cpu>,
        max_len: usize,
    ) -> Result<Tensor<usize, 2, Cpu>> {
        let mut current_ids = prompt_ids.clone();
        let [batch_size, _] = *current_ids.shape();

        for _ in 0..max_len {
            let logits = self.forward(&current_ids)?;
            // Get logits for the last token: [Batch, Seq, Vocab] -> [Batch, 1, Vocab]
            let [_, seq_len, vocab_size] = *logits.shape();

            // We want the last token's logits.
            // Since we don't have slicing yet, we might have to be creative.
            // Or just implement slicing/gather.
            // For now, let's iterate over data manually to find max.
            // This is slow but works for CPU demo.

            let data = logits.data();
            let mut next_tokens = Vec::with_capacity(batch_size);

            for b in 0..batch_size {
                let start = b * seq_len * vocab_size + (seq_len - 1) * vocab_size;
                let end = start + vocab_size;
                let last_token_logits = &data[start..end];

                // Argmax
                let mut max_val = T::min_value();
                let mut max_idx = 0;
                for (i, &val) in last_token_logits.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }
                next_tokens.push(max_idx);
            }

            // Append next_tokens to current_ids
            // Need to reconstruct tensor.
            // This flattens [Batch, Seq]. We need to insert correctly?
            // No, [Batch, Seq] is contiguous. But adding a column is tricky in flat buffer.
            // [B0_S0, B0_S1, ..., B1_S0, ...]
            // We need [B0_S0, ..., B0_New, B1_S0, ..., B1_New]
            // So we need to rebuild.

            let mut new_data = Vec::with_capacity(batch_size * (seq_len + 1));
            let old_data = current_ids.data();
            for (b, &token) in next_tokens.iter().enumerate() {
                let start = b * seq_len;
                let end = start + seq_len;
                new_data.extend_from_slice(&old_data[start..end]);
                new_data.push(token);
            }

            current_ids = Tensor::new(new_data, [batch_size, seq_len + 1])?;
        }

        Ok(current_ids)
    }

    fn load_weights<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        Self::load_weights_with_config(path, GemmaConfig::gemma_2b())
    }
}

impl<T: TensorElem + Float> GemmaForCausalLM<T> {
    pub fn load_weights_with_config<P: AsRef<std::path::Path>>(
        path: P,
        config: GemmaConfig,
    ) -> Result<Self> {
        let file =
            File::open(path).map_err(|e| crate::tensor::TensorError::Unsupported(e.to_string()))?;
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| crate::tensor::TensorError::Unsupported(e.to_string()))?
        };
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| crate::tensor::TensorError::Unsupported(e.to_string()))?;

        // Helper to load a tensor
        let load = |name: &str, shape: &[usize]| -> Result<Tensor<T, 2, Cpu>> {
            let view = tensors.tensor(name).map_err(|e| {
                crate::tensor::TensorError::Unsupported(format!("Missing tensor {}: {}", name, e))
            })?;
            let data_bytes = view.data();

            // Assume f32 for now. TODO: Handle bf16
            // If T is f32, we can cast.
            // This is unsafe if alignment/endianness is wrong, but standard for this context.
            let data: Vec<T> = if std::mem::size_of::<T>() == 4 {
                // f32
                let f32_data: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        data_bytes.as_ptr() as *const f32,
                        data_bytes.len() / 4,
                    )
                };
                f32_data.iter().map(|&x| T::from_f32(x).unwrap()).collect()
            } else {
                return Err(crate::tensor::TensorError::Unsupported(
                    "Only f32 supported for loading".into(),
                ));
            };

            Tensor::new(data, shape.try_into().unwrap())
        };

        // Load Embeddings
        let embed_w = load(
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
        let embed_tokens = Embedding::new(embed_w.clone());

        // Load Layers
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);

            // Attention
            let q_proj = Linear::new(
                load(
                    &format!("{}.self_attn.q_proj.weight", prefix),
                    &[
                        config.num_attention_heads * config.head_dim,
                        config.hidden_size,
                    ],
                )?,
                None,
            );
            let k_proj = Linear::new(
                load(
                    &format!("{}.self_attn.k_proj.weight", prefix),
                    &[
                        config.num_key_value_heads * config.head_dim,
                        config.hidden_size,
                    ],
                )?,
                None,
            );
            let v_proj = Linear::new(
                load(
                    &format!("{}.self_attn.v_proj.weight", prefix),
                    &[
                        config.num_key_value_heads * config.head_dim,
                        config.hidden_size,
                    ],
                )?,
                None,
            );
            let o_proj = Linear::new(
                load(
                    &format!("{}.self_attn.o_proj.weight", prefix),
                    &[
                        config.hidden_size,
                        config.num_attention_heads * config.head_dim,
                    ],
                )?,
                None,
            );

            let self_attn = MultiHeadAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
            );

            // MLP
            let gate_proj = Linear::new(
                load(
                    &format!("{}.mlp.gate_proj.weight", prefix),
                    &[config.intermediate_size, config.hidden_size],
                )?,
                None,
            );
            let up_proj = Linear::new(
                load(
                    &format!("{}.mlp.up_proj.weight", prefix),
                    &[config.intermediate_size, config.hidden_size],
                )?,
                None,
            );
            let down_proj = Linear::new(
                load(
                    &format!("{}.mlp.down_proj.weight", prefix),
                    &[config.hidden_size, config.intermediate_size],
                )?,
                None,
            );

            let mlp = MLP {
                gate_proj,
                up_proj,
                down_proj,
            };

            let input_layernorm_w = load(
                &format!("{}.input_layernorm.weight", prefix),
                &[config.hidden_size, 1],
            )?
            .reshape([config.hidden_size])?;
            let post_attention_layernorm_w = load(
                &format!("{}.post_attention_layernorm.weight", prefix),
                &[config.hidden_size, 1],
            )?
            .reshape([config.hidden_size])?;

            let input_layernorm =
                RMSNorm::new(input_layernorm_w, T::from_f32(config.rms_norm_eps).unwrap());
            let post_attention_layernorm = RMSNorm::new(
                post_attention_layernorm_w,
                T::from_f32(config.rms_norm_eps).unwrap(),
            );

            layers.push(GemmaBlock {
                self_attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            });
        }

        // Final Norm
        let norm_w =
            load("model.norm.weight", &[config.hidden_size, 1])?.reshape([config.hidden_size])?;
        let norm = RMSNorm::new(norm_w, T::from_f32(config.rms_norm_eps).unwrap());

        let model = GemmaModel { layers, norm };

        // LM Head
        let lm_head_w = if tensors.tensor("lm_head.weight").is_ok() {
            load("lm_head.weight", &[config.vocab_size, config.hidden_size])?
        } else {
            embed_w.clone()
        };
        let lm_head = Linear::new(lm_head_w, None);

        Ok(Self {
            model,
            lm_head,
            embed_tokens,
            config,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::transformer::attention::MultiHeadAttention;
    use crate::nn::{Embedding, Linear, RMSNorm};
    use crate::tensor::Tensor;

    #[test]
    fn test_gemma_causal_lm_forward() {
        // Use tiny config for testing
        let config = GemmaConfig::tiny_test();
        let hidden_dim = config.hidden_size;
        let vocab_size = config.vocab_size;

        // Create dummy weights
        let embed_w = Tensor::zeros([vocab_size, hidden_dim]);
        let embed = Embedding::new(embed_w);

        let lm_head_w = Tensor::zeros([vocab_size, hidden_dim]);
        let lm_head = Linear::new(lm_head_w, None);

        let norm_w = Tensor::ones([hidden_dim]);
        let norm = RMSNorm::new(norm_w, config.rms_norm_eps);

        // Create one block
        let q_proj = Linear::new(Tensor::zeros([hidden_dim, hidden_dim]), None);
        let k_proj = Linear::new(
            Tensor::zeros([config.num_key_value_heads * config.head_dim, hidden_dim]),
            None,
        );
        let v_proj = Linear::new(
            Tensor::zeros([config.num_key_value_heads * config.head_dim, hidden_dim]),
            None,
        );
        let o_proj = Linear::new(Tensor::zeros([hidden_dim, hidden_dim]), None);

        let attn = MultiHeadAttention::new(
            hidden_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        );

        let gate_proj = Linear::new(Tensor::zeros([config.intermediate_size, hidden_dim]), None);
        let up_proj = Linear::new(Tensor::zeros([config.intermediate_size, hidden_dim]), None);
        let down_proj = Linear::new(Tensor::zeros([hidden_dim, config.intermediate_size]), None);
        let mlp = MLP {
            gate_proj,
            up_proj,
            down_proj,
        };

        let block = GemmaBlock {
            self_attn: attn,
            mlp,
            input_layernorm: RMSNorm::new(Tensor::ones([hidden_dim]), config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(Tensor::ones([hidden_dim]), config.rms_norm_eps),
        };

        let model = GemmaModel {
            layers: vec![block], // Just 1 layer for test
            norm,
        };

        let gemma = GemmaForCausalLM {
            model,
            lm_head,
            embed_tokens: embed,
            config: config.clone(),
        };

        // Input: [Batch=1, Seq=2]
        let input_ids = Tensor::new(vec![0, 1], [1, 2]).unwrap();

        let output = gemma.forward(&input_ids).unwrap();

        assert_eq!(output.shape(), &[1, 2, vocab_size]);
    }

    #[test]
    fn test_gemma_generate() {
        // Use tiny config
        let config = GemmaConfig::tiny_test();
        let hidden_dim = config.hidden_size;
        let vocab_size = config.vocab_size;

        // Create dummy weights
        let embed_w = Tensor::zeros([vocab_size, hidden_dim]);
        let embed = Embedding::new(embed_w);

        let lm_head_w = Tensor::zeros([vocab_size, hidden_dim]);
        let lm_head = Linear::new(lm_head_w, None);

        let norm_w = Tensor::ones([hidden_dim]);
        let norm = RMSNorm::new(norm_w, config.rms_norm_eps);

        // Create one block (identity-ish)
        let q_proj = Linear::new(Tensor::zeros([hidden_dim, hidden_dim]), None);
        let k_proj = Linear::new(
            Tensor::zeros([config.num_key_value_heads * config.head_dim, hidden_dim]),
            None,
        );
        let v_proj = Linear::new(
            Tensor::zeros([config.num_key_value_heads * config.head_dim, hidden_dim]),
            None,
        );
        let o_proj = Linear::new(Tensor::zeros([hidden_dim, hidden_dim]), None);

        let attn = MultiHeadAttention::new(
            hidden_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        );

        let gate_proj = Linear::new(Tensor::zeros([config.intermediate_size, hidden_dim]), None);
        let up_proj = Linear::new(Tensor::zeros([config.intermediate_size, hidden_dim]), None);
        let down_proj = Linear::new(Tensor::zeros([hidden_dim, config.intermediate_size]), None);
        let mlp = MLP {
            gate_proj,
            up_proj,
            down_proj,
        };

        let block = GemmaBlock {
            self_attn: attn,
            mlp,
            input_layernorm: RMSNorm::new(Tensor::ones([hidden_dim]), config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(Tensor::ones([hidden_dim]), config.rms_norm_eps),
        };

        let model = GemmaModel {
            layers: vec![block],
            norm,
        };

        let gemma = GemmaForCausalLM {
            model,
            lm_head,
            embed_tokens: embed,
            config: config.clone(),
        };

        // Input: [Batch=1, Seq=2]
        let input_ids = Tensor::new(vec![0, 1], [1, 2]).unwrap();

        // Generate 3 tokens
        let output_ids = gemma.generate(&input_ids, 3).unwrap();

        // Output should be [1, 2 + 3] = [1, 5]
        assert_eq!(output_ids.shape(), &[1, 5]);

        // Since weights are zero, logits are zero (or constant).
        // Argmax of zeros is 0.
        // So generated tokens should be 0.
        let data = output_ids.data();
        assert_eq!(data[2], 0);
        assert_eq!(data[3], 0);
        assert_eq!(data[4], 0);
    }

    #[test]
    fn test_config_defaults() {
        let c70b = GemmaConfig::gemma_70b();
        assert_eq!(c70b.hidden_size, 8192);
        assert_eq!(c70b.num_hidden_layers, 80);

        let c2b = GemmaConfig::gemma_2b();
        assert_eq!(c2b.hidden_size, 2048);
        assert_eq!(c2b.num_hidden_layers, 18);
    }

    #[test]
    fn test_load_weights_unsupported() {
        // Test loading from a non-existent file
        let result = GemmaForCausalLM::<f32>::load_weights("non_existent_file.safetensors");
        assert!(result.is_err());
        match result {
            Err(crate::tensor::TensorError::Unsupported(_)) => (), // Expected
            _ => panic!("Expected Unsupported error"),
        }
    }

    #[test]
    fn test_load_weights_success() {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::HashMap;
        use std::fs::File;
        use std::io::Write;

        let config = GemmaConfig::tiny_test();
        let path = "test_gemma_weights.safetensors";

        // Create dummy data
        let mut tensors = HashMap::new();

        // Helper to add tensor
        let mut add_tensor = |name: String, shape: Vec<usize>| {
            let size: usize = shape.iter().product();
            let data: Vec<u8> = vec![0u8; size * 4]; // f32 zeros
            tensors.insert(name, (Dtype::F32, shape, data));
        };

        // Embeddings
        add_tensor(
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        );

        // Layers
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            add_tensor(
                format!("{}.self_attn.q_proj.weight", prefix),
                vec![
                    config.num_attention_heads * config.head_dim,
                    config.hidden_size,
                ],
            );
            add_tensor(
                format!("{}.self_attn.k_proj.weight", prefix),
                vec![
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                ],
            );
            add_tensor(
                format!("{}.self_attn.v_proj.weight", prefix),
                vec![
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                ],
            );
            add_tensor(
                format!("{}.self_attn.o_proj.weight", prefix),
                vec![
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim,
                ],
            );

            add_tensor(
                format!("{}.mlp.gate_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            );
            add_tensor(
                format!("{}.mlp.up_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            );
            add_tensor(
                format!("{}.mlp.down_proj.weight", prefix),
                vec![config.hidden_size, config.intermediate_size],
            );

            add_tensor(
                format!("{}.input_layernorm.weight", prefix),
                vec![config.hidden_size, 1],
            );
            add_tensor(
                format!("{}.post_attention_layernorm.weight", prefix),
                vec![config.hidden_size, 1],
            );
        }

        // Final Norm
        add_tensor("model.norm.weight".to_string(), vec![config.hidden_size, 1]);

        // LM Head (optional, we can skip it to test fallback, or add it. Let's add it)
        add_tensor(
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        );

        // Serialize
        // We need to convert HashMap to what safetensors expects.
        // safetensors::serialize takes a generic map.
        // We need to construct TensorViews.

        let views: HashMap<String, TensorView> = tensors
            .iter()
            .map(|(k, (dtype, shape, data))| {
                (
                    k.clone(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        let serialized = safetensors::serialize(&views, None).unwrap();

        let mut file = File::create(path).unwrap();
        file.write_all(&serialized).unwrap();

        // Test loading
        let model =
            GemmaForCausalLM::<f32>::load_weights_with_config(path, config.clone()).unwrap();

        assert_eq!(model.config.hidden_size, config.hidden_size);
        assert_eq!(model.model.layers.len(), config.num_hidden_layers);

        // Cleanup
        std::fs::remove_file(path).unwrap();
    }
}
