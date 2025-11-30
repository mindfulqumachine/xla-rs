use crate::models::traits::CausalLM;
use crate::nn::transformer::attention::MultiHeadAttention;
use crate::nn::{Activation, Embedding, LayerNorm, Linear};
use crate::tensor::{Cpu, Result, Tensor, TensorElem, TensorOps};
use memmap2::Mmap;
use num_traits::Float;
use safetensors::SafeTensors;
use std::fs::File;
use std::ops::Add;

#[derive(Debug, Clone)]
pub struct GPT2Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub layer_norm_epsilon: f32,
}

impl GPT2Config {
    pub fn gpt2() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            layer_norm_epsilon: 1e-5,
        }
    }
}

#[derive(Debug)]
pub struct GPT2MLP<T: TensorElem> {
    pub c_fc: Linear<T>,
    pub c_proj: Linear<T>,
}

impl<T: TensorElem + Float> GPT2MLP<T> {
    pub fn forward(&self, x: &Tensor<T, 3, Cpu>) -> Result<Tensor<T, 3, Cpu>> {
        let x = self.c_fc.forward(x)?;
        let x = Activation::gelu(&x);
        self.c_proj.forward(&x)
    }
}

#[derive(Debug)]
pub struct GPT2Block<T: TensorElem> {
    pub ln_1: LayerNorm<T>,
    pub attn: MultiHeadAttention<T>,
    pub ln_2: LayerNorm<T>,
    pub mlp: GPT2MLP<T>,
}

impl<T: TensorElem + Float> GPT2Block<T> {
    pub fn forward(
        &self,
        x: &Tensor<T, 3, Cpu>,
        mask: Option<&Tensor<T, 2, Cpu>>,
    ) -> Result<Tensor<T, 3, Cpu>> {
        let residual = x;
        let norm_x = self.ln_1.forward(x)?;

        // GPT-2 uses standard attention without RoPE.
        // We pass None for freqs.
        let attn_out = self.attn.forward(&norm_x, None, None, mask)?;

        let x = residual.add(&attn_out)?;

        let residual = &x;
        let norm_x = self.ln_2.forward(&x)?;
        let mlp_out = self.mlp.forward(&norm_x)?;

        let x = residual.add(&mlp_out)?;

        Ok(x)
    }
}

#[derive(Debug)]
pub struct GPT2Model<T: TensorElem> {
    pub wte: Embedding<T>,
    pub wpe: Embedding<T>,
    pub h: Vec<GPT2Block<T>>,
    pub ln_f: LayerNorm<T>,
}

impl<T: TensorElem + Float> GPT2Model<T> {
    pub fn forward(
        &self,
        input_ids: &Tensor<usize, 2, Cpu>,
        mask: Option<&Tensor<T, 2, Cpu>>,
    ) -> Result<Tensor<T, 3, Cpu>> {
        let [batch_size, seq_len] = *input_ids.shape();

        // Embeddings
        let inputs_embeds = self.wte.forward(input_ids)?;

        // Position Embeddings
        // Create position ids [0, 1, ..., seq_len-1]
        let mut pos_ids_vec = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            pos_ids_vec.push(i);
        }
        // Repeat for batch
        let mut pos_ids_batch = Vec::with_capacity(batch_size * seq_len);
        for _ in 0..batch_size {
            pos_ids_batch.extend_from_slice(&pos_ids_vec);
        }
        let position_ids = Tensor::new(pos_ids_batch, [batch_size, seq_len])?;
        let position_embeds = self.wpe.forward(&position_ids)?;

        let mut hidden_states = inputs_embeds.add(&position_embeds)?;

        for block in &self.h {
            hidden_states = block.forward(&hidden_states, mask)?;
        }

        self.ln_f.forward(&hidden_states)
    }
}

#[derive(Debug)]
pub struct GPT2LMHeadModel<T: TensorElem> {
    pub transformer: GPT2Model<T>,
    pub lm_head: Linear<T>,
    pub config: GPT2Config,
}

impl<T: TensorElem + Float> CausalLM<T> for GPT2LMHeadModel<T> {
    fn forward(&self, input_ids: &Tensor<usize, 2, Cpu>) -> Result<Tensor<T, 3, Cpu>> {
        let hidden_states = self.transformer.forward(input_ids, None)?;
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
            let [_, seq_len, vocab_size] = *logits.shape();

            let data = logits.data();
            let mut next_tokens = Vec::with_capacity(batch_size);

            for b in 0..batch_size {
                let start = b * seq_len * vocab_size + (seq_len - 1) * vocab_size;
                let end = start + vocab_size;
                let last_token_logits = &data[start..end];

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
        Self::load_weights_with_config(path, GPT2Config::gpt2())
    }
}

impl<T: TensorElem + Float> GPT2LMHeadModel<T> {
    pub fn load_weights_with_config<P: AsRef<std::path::Path>>(
        path: P,
        config: GPT2Config,
    ) -> Result<Self> {
        let file =
            File::open(path).map_err(|e| crate::tensor::TensorError::Unsupported(e.to_string()))?;
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| crate::tensor::TensorError::Unsupported(e.to_string()))?
        };
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| crate::tensor::TensorError::Unsupported(e.to_string()))?;

        let load_data = |name: &str| -> Result<Vec<T>> {
            let view = tensors.tensor(name).map_err(|e| {
                crate::tensor::TensorError::Unsupported(format!("Missing tensor {}: {}", name, e))
            })?;
            let data_bytes = view.data();

            if std::mem::size_of::<T>() == 4 {
                let f32_data: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        data_bytes.as_ptr() as *const f32,
                        data_bytes.len() / 4,
                    )
                };
                Ok(f32_data.iter().map(|&x| T::from_f32(x).unwrap()).collect())
            } else {
                Err(crate::tensor::TensorError::Unsupported(
                    "Only f32 supported for loading".into(),
                ))
            }
        };

        let load = |name: &str, shape: &[usize]| -> Result<Tensor<T, 2, Cpu>> {
            let data = load_data(name)?;
            Tensor::new(data, shape.try_into().unwrap())
        };

        let load_1d = |name: &str, shape: &[usize]| -> Result<Tensor<T, 1, Cpu>> {
            let data = load_data(name)?;
            Tensor::new(data, shape.try_into().unwrap())
        };

        // Load Embeddings
        let wte_w = load("wte.weight", &[config.vocab_size, config.n_embd])?;
        let wte = Embedding::new(wte_w.clone());

        let wpe_w = load("wpe.weight", &[config.n_positions, config.n_embd])?;
        let wpe = Embedding::new(wpe_w);

        let mut h = Vec::new();
        for i in 0..config.n_layer {
            let prefix = format!("h.{}", i);

            // LayerNorms
            let ln_1_w = load_1d(&format!("{}.ln_1.weight", prefix), &[config.n_embd])?;
            let ln_1_b = load_1d(&format!("{}.ln_1.bias", prefix), &[config.n_embd])?;
            let ln_1 = LayerNorm::new(
                ln_1_w,
                ln_1_b,
                T::from_f32(config.layer_norm_epsilon).unwrap(),
            );

            let ln_2_w = load_1d(&format!("{}.ln_2.weight", prefix), &[config.n_embd])?;
            let ln_2_b = load_1d(&format!("{}.ln_2.bias", prefix), &[config.n_embd])?;
            let ln_2 = LayerNorm::new(
                ln_2_w,
                ln_2_b,
                T::from_f32(config.layer_norm_epsilon).unwrap(),
            );

            // Attention
            // c_attn weight is [n_embd, 3 * n_embd] (Conv1D in HF is [in, out])
            // Wait, HF Conv1D weight is [in, out]. PyTorch Linear is [out, in].
            // xla-rs Linear is [out, in].
            // If we load from HF safetensors, we need to check if it's transposed.
            // Usually HF safetensors for GPT-2 are from PyTorch, so Linear weights are [out, in].
            // But GPT-2 uses Conv1D which is [in, out].
            // Let's assume standard PyTorch Linear layout for simplicity or handle transpose.
            // Actually, GPT-2 implementation in transformers uses Conv1D.
            // "w" is [nx, nf].
            // If we assume we are loading converted weights or standard Linear weights.
            // Let's assume [3*n_embd, n_embd] for c_attn.weight if it was Linear.
            // If it is Conv1D, it is [n_embd, 3*n_embd].
            // Let's try to load as [n_embd, 3*n_embd] and transpose if needed.
            // But `load` expects shape.
            // Let's assume we load [n_embd, 3*n_embd] and then split.

            let c_attn_w = load(
                &format!("{}.attn.c_attn.weight", prefix),
                &[config.n_embd, 3 * config.n_embd],
            )?;
            let c_attn_b = load_1d(
                &format!("{}.attn.c_attn.bias", prefix),
                &[3 * config.n_embd],
            )?;

            // We need to split c_attn into q, k, v.
            // c_attn is [n_embd, 3*n_embd].
            // We want [n_embd, n_embd] for each.
            // And we need to transpose to [n_embd, n_embd] (out, in) for Linear?
            // xla-rs Linear expects [out_features, in_features].
            // So we need [n_embd, n_embd].
            // If c_attn is [in, out], then we need to transpose it to [out, in] -> [3*n_embd, n_embd].
            // Then split along dim 0.

            let c_attn_w_t = c_attn_w.transpose()?; // [3*n_embd, n_embd]

            // Split is not implemented in Tensor yet?
            // We can slice manually.
            // Or implement split.
            // Let's slice manually using data.

            let split_weights = |w: &Tensor<T, 2, Cpu>,
                                 b: &Tensor<T, 1, Cpu>|
             -> Result<(Linear<T>, Linear<T>, Linear<T>)> {
                let w_data = w.data();
                let b_data = b.data();
                let size = config.n_embd * config.n_embd;

                let q_w_data = w_data[0..size].to_vec();
                let k_w_data = w_data[size..2 * size].to_vec();
                let v_w_data = w_data[2 * size..3 * size].to_vec();

                let q_b_data = b_data[0..config.n_embd].to_vec();
                let k_b_data = b_data[config.n_embd..2 * config.n_embd].to_vec();
                let v_b_data = b_data[2 * config.n_embd..3 * config.n_embd].to_vec();

                let q_proj = Linear::new(
                    Tensor::new(q_w_data, [config.n_embd, config.n_embd])?,
                    Some(Tensor::new(q_b_data, [config.n_embd])?),
                );
                let k_proj = Linear::new(
                    Tensor::new(k_w_data, [config.n_embd, config.n_embd])?,
                    Some(Tensor::new(k_b_data, [config.n_embd])?),
                );
                let v_proj = Linear::new(
                    Tensor::new(v_w_data, [config.n_embd, config.n_embd])?,
                    Some(Tensor::new(v_b_data, [config.n_embd])?),
                );
                Ok((q_proj, k_proj, v_proj))
            };

            let (q_proj, k_proj, v_proj) = split_weights(&c_attn_w_t, &c_attn_b)?;

            let c_proj_w = load(
                &format!("{}.attn.c_proj.weight", prefix),
                &[config.n_embd, config.n_embd],
            )?;
            let c_proj_b = load_1d(&format!("{}.attn.c_proj.bias", prefix), &[config.n_embd])?;
            let o_proj = Linear::new(c_proj_w.transpose()?, Some(c_proj_b));

            let attn = MultiHeadAttention::new(
                config.n_embd,
                config.n_head,
                config.n_head, // KV heads = heads
                config.n_embd / config.n_head,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
            );

            // MLP
            let c_fc_w = load(
                &format!("{}.mlp.c_fc.weight", prefix),
                &[config.n_embd, 4 * config.n_embd],
            )?;
            let c_fc_b = load_1d(&format!("{}.mlp.c_fc.bias", prefix), &[4 * config.n_embd])?;
            let c_fc = Linear::new(c_fc_w.transpose()?, Some(c_fc_b));

            let c_proj_mlp_w = load(
                &format!("{}.mlp.c_proj.weight", prefix),
                &[4 * config.n_embd, config.n_embd],
            )?;
            let c_proj_mlp_b = load_1d(&format!("{}.mlp.c_proj.bias", prefix), &[config.n_embd])?;
            let c_proj = Linear::new(c_proj_mlp_w.transpose()?, Some(c_proj_mlp_b));

            let mlp = GPT2MLP { c_fc, c_proj };

            h.push(GPT2Block {
                ln_1,
                attn,
                ln_2,
                mlp,
            });
        }

        let ln_f_w = load_1d("ln_f.weight", &[config.n_embd])?;
        let ln_f_b = load_1d("ln_f.bias", &[config.n_embd])?;
        let ln_f = LayerNorm::new(
            ln_f_w,
            ln_f_b,
            T::from_f32(config.layer_norm_epsilon).unwrap(),
        );

        let transformer = GPT2Model {
            wte: wte.clone(),
            wpe,
            h,
            ln_f,
        };

        // LM Head (tied to wte)
        let lm_head = Linear::new(wte_w.transpose()?, None); // Usually bias is None? Or tied?
        // GPT-2 usually doesn't have a separate bias for lm_head, it uses the embedding weights.

        Ok(Self {
            transformer,
            lm_head,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_gpt2_config_defaults() {
        let config = GPT2Config::gpt2();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.n_positions, 1024);
        assert_eq!(config.n_embd, 768);
        assert_eq!(config.n_layer, 12);
        assert_eq!(config.n_head, 12);
        assert_eq!(config.layer_norm_epsilon, 1e-5);
    }

    fn create_dummy_config() -> GPT2Config {
        GPT2Config {
            vocab_size: 100,
            n_positions: 20,
            n_embd: 32,
            n_layer: 2,
            n_head: 4,
            layer_norm_epsilon: 1e-5,
        }
    }

    #[test]
    fn test_gpt2_mlp_forward() {
        let config = create_dummy_config();
        let c_fc = Linear::new(
            Tensor::zeros([4 * config.n_embd, config.n_embd]),
            Some(Tensor::zeros([4 * config.n_embd])),
        );
        let c_proj = Linear::new(
            Tensor::zeros([config.n_embd, 4 * config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let mlp = GPT2MLP { c_fc, c_proj };

        let input = Tensor::<f32, 3, Cpu>::zeros([1, 5, config.n_embd]);
        let output = mlp.forward(&input).unwrap();
        assert_eq!(*output.shape(), [1, 5, config.n_embd]);
    }

    #[test]
    fn test_gpt2_block_forward() {
        let config = create_dummy_config();

        // Setup dummy components
        let ln_1 = LayerNorm::new(
            Tensor::ones([config.n_embd]),
            Tensor::zeros([config.n_embd]),
            1e-5,
        );
        let ln_2 = LayerNorm::new(
            Tensor::ones([config.n_embd]),
            Tensor::zeros([config.n_embd]),
            1e-5,
        );

        let head_dim = config.n_embd / config.n_head;
        let q_proj = Linear::new(
            Tensor::zeros([config.n_embd, config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let k_proj = Linear::new(
            Tensor::zeros([config.n_embd, config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let v_proj = Linear::new(
            Tensor::zeros([config.n_embd, config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let o_proj = Linear::new(
            Tensor::zeros([config.n_embd, config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );

        let attn = MultiHeadAttention::new(
            config.n_embd,
            config.n_head,
            config.n_head,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        );

        let c_fc = Linear::new(
            Tensor::zeros([4 * config.n_embd, config.n_embd]),
            Some(Tensor::zeros([4 * config.n_embd])),
        );
        let c_proj = Linear::new(
            Tensor::zeros([config.n_embd, 4 * config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let mlp = GPT2MLP { c_fc, c_proj };

        let block = GPT2Block {
            ln_1,
            attn,
            ln_2,
            mlp,
        };

        let input = Tensor::<f32, 3, Cpu>::zeros([1, 5, config.n_embd]);
        let output = block.forward(&input, None).unwrap();
        assert_eq!(*output.shape(), [1, 5, config.n_embd]);
    }

    #[test]
    fn test_gpt2_model_forward() {
        let config = create_dummy_config();

        let wte = Embedding::new(Tensor::zeros([config.vocab_size, config.n_embd]));
        let wpe = Embedding::new(Tensor::zeros([config.n_positions, config.n_embd]));
        let ln_f = LayerNorm::new(
            Tensor::ones([config.n_embd]),
            Tensor::zeros([config.n_embd]),
            1e-5,
        );

        // Create one block
        let head_dim = config.n_embd / config.n_head;
        let q_proj = Linear::new(
            Tensor::zeros([config.n_embd, config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let k_proj = Linear::new(
            Tensor::zeros([config.n_embd, config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let v_proj = Linear::new(
            Tensor::zeros([config.n_embd, config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let o_proj = Linear::new(
            Tensor::zeros([config.n_embd, config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let attn = MultiHeadAttention::new(
            config.n_embd,
            config.n_head,
            config.n_head,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        );
        let c_fc = Linear::new(
            Tensor::zeros([4 * config.n_embd, config.n_embd]),
            Some(Tensor::zeros([4 * config.n_embd])),
        );
        let c_proj = Linear::new(
            Tensor::zeros([config.n_embd, 4 * config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let mlp = GPT2MLP { c_fc, c_proj };
        let block = GPT2Block {
            ln_1: LayerNorm::new(
                Tensor::ones([config.n_embd]),
                Tensor::zeros([config.n_embd]),
                1e-5,
            ),
            attn,
            ln_2: LayerNorm::new(
                Tensor::ones([config.n_embd]),
                Tensor::zeros([config.n_embd]),
                1e-5,
            ),
            mlp,
        };

        let model = GPT2Model {
            wte,
            wpe,
            h: vec![block],
            ln_f,
        };

        let input_ids = Tensor::new(vec![0usize, 1, 2, 3, 4], [1, 5]).unwrap();
        let output = model.forward(&input_ids, None).unwrap();
        assert_eq!(*output.shape(), [1, 5, config.n_embd]);
    }

    #[test]
    fn test_gpt2_lm_head_model_forward_and_generate() {
        let config = create_dummy_config();

        // Construct minimal model
        let wte = Embedding::new(Tensor::zeros([config.vocab_size, config.n_embd]));
        let wpe = Embedding::new(Tensor::zeros([config.n_positions, config.n_embd]));
        let ln_f = LayerNorm::new(
            Tensor::ones([config.n_embd]),
            Tensor::zeros([config.n_embd]),
            1e-5,
        );
        let head_dim = config.n_embd / config.n_head;
        let q_proj = Linear::new(
            Tensor::zeros([config.n_embd, config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let k_proj = Linear::new(
            Tensor::zeros([config.n_embd, config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let v_proj = Linear::new(
            Tensor::zeros([config.n_embd, config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let o_proj = Linear::new(
            Tensor::zeros([config.n_embd, config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let attn = MultiHeadAttention::new(
            config.n_embd,
            config.n_head,
            config.n_head,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        );
        let c_fc = Linear::new(
            Tensor::zeros([4 * config.n_embd, config.n_embd]),
            Some(Tensor::zeros([4 * config.n_embd])),
        );
        let c_proj = Linear::new(
            Tensor::zeros([config.n_embd, 4 * config.n_embd]),
            Some(Tensor::zeros([config.n_embd])),
        );
        let mlp = GPT2MLP { c_fc, c_proj };
        let block = GPT2Block {
            ln_1: LayerNorm::new(
                Tensor::ones([config.n_embd]),
                Tensor::zeros([config.n_embd]),
                1e-5,
            ),
            attn,
            ln_2: LayerNorm::new(
                Tensor::ones([config.n_embd]),
                Tensor::zeros([config.n_embd]),
                1e-5,
            ),
            mlp,
        };
        let transformer = GPT2Model {
            wte: wte.clone(),
            wpe,
            h: vec![block],
            ln_f,
        };
        let lm_head = Linear::new(Tensor::zeros([config.vocab_size, config.n_embd]), None);

        let model = GPT2LMHeadModel {
            transformer,
            lm_head,
            config: config.clone(),
        };

        let input_ids = Tensor::new(vec![0usize, 1], [1, 2]).unwrap();
        let logits = model.forward(&input_ids).unwrap();
        assert_eq!(*logits.shape(), [1, 2, config.vocab_size]);

        // Test generate
        let generated = model.generate(&input_ids, 2).unwrap();
        assert_eq!(*generated.shape(), [1, 4]); // 2 input + 2 generated
    }

    #[test]
    fn test_load_weights() {
        let config = create_dummy_config();

        // Create a temporary safetensors file
        let file = NamedTempFile::new().unwrap();

        // Helper to create dummy data
        let create_tensor_data = |shape: &[usize]| -> Vec<u8> {
            let num_elements: usize = shape.iter().product();
            let data: Vec<f32> = (0..num_elements).map(|i| i as f32 * 0.01).collect();
            let mut bytes = Vec::with_capacity(num_elements * 4);
            for f in data {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
            bytes
        };

        // Prepare tensors
        let wte_shape = [config.vocab_size, config.n_embd];
        let wpe_shape = [config.n_positions, config.n_embd];
        let ln_shape = [config.n_embd];
        // c_attn is [n_embd, 3*n_embd] in Conv1D (in, out)
        let c_attn_shape = [config.n_embd, 3 * config.n_embd];
        let c_attn_bias_shape = [3 * config.n_embd];
        let c_proj_shape = [config.n_embd, config.n_embd];
        let c_fc_shape = [config.n_embd, 4 * config.n_embd];
        let c_fc_bias_shape = [4 * config.n_embd];
        let c_proj_mlp_shape = [4 * config.n_embd, config.n_embd];

        let tensors = vec![
            (
                "wte.weight",
                wte_shape.as_slice(),
                create_tensor_data(&wte_shape),
            ),
            (
                "wpe.weight",
                wpe_shape.as_slice(),
                create_tensor_data(&wpe_shape),
            ),
            (
                "ln_f.weight",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
            (
                "ln_f.bias",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
            // Layer 0
            (
                "h.0.ln_1.weight",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
            (
                "h.0.ln_1.bias",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
            (
                "h.0.ln_2.weight",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
            (
                "h.0.ln_2.bias",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
            (
                "h.0.attn.c_attn.weight",
                c_attn_shape.as_slice(),
                create_tensor_data(&c_attn_shape),
            ),
            (
                "h.0.attn.c_attn.bias",
                c_attn_bias_shape.as_slice(),
                create_tensor_data(&c_attn_bias_shape),
            ),
            (
                "h.0.attn.c_proj.weight",
                c_proj_shape.as_slice(),
                create_tensor_data(&c_proj_shape),
            ),
            (
                "h.0.attn.c_proj.bias",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
            (
                "h.0.mlp.c_fc.weight",
                c_fc_shape.as_slice(),
                create_tensor_data(&c_fc_shape),
            ),
            (
                "h.0.mlp.c_fc.bias",
                c_fc_bias_shape.as_slice(),
                create_tensor_data(&c_fc_bias_shape),
            ),
            (
                "h.0.mlp.c_proj.weight",
                c_proj_mlp_shape.as_slice(),
                create_tensor_data(&c_proj_mlp_shape),
            ),
            (
                "h.0.mlp.c_proj.bias",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
            // Layer 1
            (
                "h.1.ln_1.weight",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
            (
                "h.1.ln_1.bias",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
            (
                "h.1.ln_2.weight",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
            (
                "h.1.ln_2.bias",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
            (
                "h.1.attn.c_attn.weight",
                c_attn_shape.as_slice(),
                create_tensor_data(&c_attn_shape),
            ),
            (
                "h.1.attn.c_attn.bias",
                c_attn_bias_shape.as_slice(),
                create_tensor_data(&c_attn_bias_shape),
            ),
            (
                "h.1.attn.c_proj.weight",
                c_proj_shape.as_slice(),
                create_tensor_data(&c_proj_shape),
            ),
            (
                "h.1.attn.c_proj.bias",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
            (
                "h.1.mlp.c_fc.weight",
                c_fc_shape.as_slice(),
                create_tensor_data(&c_fc_shape),
            ),
            (
                "h.1.mlp.c_fc.bias",
                c_fc_bias_shape.as_slice(),
                create_tensor_data(&c_fc_bias_shape),
            ),
            (
                "h.1.mlp.c_proj.weight",
                c_proj_mlp_shape.as_slice(),
                create_tensor_data(&c_proj_mlp_shape),
            ),
            (
                "h.1.mlp.c_proj.bias",
                ln_shape.as_slice(),
                create_tensor_data(&ln_shape),
            ),
        ];

        let data: Vec<(&str, safetensors::tensor::TensorView)> = tensors
            .iter()
            .map(|(name, shape, data)| {
                (
                    *name,
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        shape.to_vec(),
                        data.as_slice(),
                    )
                    .unwrap(),
                )
            })
            .collect();

        safetensors::serialize_to_file(data, None, file.path()).unwrap();

        let model = GPT2LMHeadModel::<f32>::load_weights_with_config(file.path(), config).unwrap();

        assert_eq!(model.transformer.h.len(), 2);
    }
}
