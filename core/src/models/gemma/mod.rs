use crate::nn::transformer::attention::MultiHeadAttention;
use crate::nn::{Activation, Linear, RMSNorm};
use crate::tensor::{Cpu, Result, Tensor, TensorElem};
use num_traits::Float;
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
            .forward(&norm_x, freqs_cos, freqs_sin, mask)?;

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
