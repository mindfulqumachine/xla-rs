# Chapter 6: The Gemma Architecture

Gemma is a family of lightweight, state-of-the-art open models from Google. In this chapter, we assemble the components from previous chapters into the full Gemma architecture.

## The Gemma Block

Each block consists of:
1.  **Input RMSNorm**
2.  **Multi-Head Attention** (with RoPE)
3.  **Post-Attention RMSNorm**
4.  **MLP** (Feed-Forward Network)

$$ x = x + \text{Attention}(\text{RMSNorm}(x)) $$
$$ x = x + \text{MLP}(\text{RMSNorm}(x)) $$

Note that Gemma uses **Pre-Norm** with a twist: the residual connection is added *after* the block.

```rust
# extern crate xla_rs;
# use xla_rs::tensor::TensorElem;
# use xla_rs::nn::transformer::attention::MultiHeadAttention;
# use xla_rs::models::gemma::MLP;
# use xla_rs::nn::RMSNorm;

pub struct GemmaBlock<T: TensorElem> {
    pub self_attn: MultiHeadAttention<T>,
    pub mlp: MLP<T>,
    pub input_layernorm: RMSNorm<T>,
    pub post_attention_layernorm: RMSNorm<T>,
}
```

## The MLP

Gemma uses a Gated Linear Unit (GLU) variant.

$$ \text{MLP}(x) = \text{Down}(\text{SiLU}(\text{Gate}(x)) \odot \text{Up}(x)) $$

## The Full Model

The `GemmaModel` is simply a stack of `GemmaBlock`s followed by a final normalization.

```rust
# extern crate xla_rs;
# use xla_rs::tensor::TensorElem;
# use xla_rs::models::gemma::GemmaBlock;
# use xla_rs::nn::RMSNorm;  // For final layer norm

pub struct GemmaModel<T: TensorElem> {
    pub layers: Vec<GemmaBlock<T>>,
    pub norm: RMSNorm<T>,
}
```
