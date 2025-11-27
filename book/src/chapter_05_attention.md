# Chapter 5: The Mechanics of Attention

Attention is the mechanism that allows the model to "focus" on different parts of the input sequence.

## Scaled Dot-Product Attention

The core operation is:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Where:
- $Q$ (Query): What I'm looking for.
- $K$ (Key): What I have.
- $V$ (Value): What I pass on.

## Rotary Positional Embeddings (RoPE)

Transformers process tokens in parallel, so they have no inherent notion of order. RoPE injects position information by rotating the Query and Key vectors in the complex plane.

$$ f_{q,k}(x_m, m) = x_m e^{im\theta} $$

In `xla-rs`, we implement this in `apply_rope`.

## Grouped Query Attention (GQA)

Gemma uses GQA, an optimization where multiple query heads share a single key/value head. This reduces memory bandwidth (KV cache size) while maintaining performance.

## Implementation

Our `MultiHeadAttention` struct handles:
1.  Projecting inputs to $Q, K, V$.
2.  Applying RoPE.
3.  Repeating KV heads (for GQA).
4.  Computing attention scores and output.

```rust
# extern crate xla_rs;
# use xla_rs::tensor::TensorElem;
# use xla_rs::nn::Linear;
pub struct MultiHeadAttention<T: TensorElem> {
    pub q_proj: Linear<T>,
    pub k_proj: Linear<T>,
    pub v_proj: Linear<T>,
    pub o_proj: Linear<T>,
    // ...
}
```
