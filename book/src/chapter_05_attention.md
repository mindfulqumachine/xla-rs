# Chapter 5: The Mechanics of Attention

> "Attention is all you need." — Vaswani et al., 2017

The Transformer architecture has revolutionized Artificial Intelligence, displacing Recurrent Neural Networks (RNNs) as the dominant architecture for sequence modeling. Unlike RNNs, which process data sequentially (and thus struggle with long-range dependencies and parallelization), Transformers process entire sequences in parallel.

At the heart of this revolution is the **Attention Mechanism**.

## The Heart of the Machine: Attention

To understand attention, let's move away from matrices for a moment and use an analogy.

### The Library Analogy

Imagine you are in a massive library. You have a specific question you want to answer.

1.  **Query (`Q`)**: This is your question. *What am I looking for?*
2.  **Key (`K`)**: Every book in the library has a catalog card or a label on its spine. This is the key. *How do I match my question to the content?*
3.  **Value (`V`)**: The content of the book itself. *What information do I actually retrieve?*

The attention mechanism works by comparing your **Query** against every **Key** in the library. The better the match between your Query and a Key, the more "attention" you pay to that book's **Value**.

![Self-Attention Vectors](images/attention_mechanism.png)
*Image Credit: [Jay Alammar](https://jalammar.github.io/illustrated-transformer/)*

```mermaid
graph LR
    subgraph User
        Q[Query: 'Question']
    end
    
    subgraph Library
        K1[Key: 'Book Label A']
        K2[Key: 'Book Label B']
        K3[Key: 'Book Label C']
        
        V1[Value: 'Content A']
        V2[Value: 'Content B']
        V3[Value: 'Content C']
    end
    
    Q -.->|Compare| K1
    Q -.->|Compare| K2
    Q -.->|Compare| K3
    
    K1 -->|Score| V1
    K2 -->|Score| V2
    K3 -->|Score| V3
    
    V1 -->|Weighted Sum| Result
    V2 -->|Weighted Sum| Result
    V3 -->|Weighted Sum| Result
```
*Diagram: The Attention Mechanism as a Library. Your Query matches against Keys to retrieve a weighted sum of Values.*

### Scaled Dot-Product Attention

Mathematically, we compute this similarity using a dot product. If two vectors point in the same direction, their dot product is large.

![Attention Matrix Calculation](images/attention_matrix.png)
*Image Credit: [Jay Alammar](https://jalammar.github.io/illustrated-transformer/)*

```mermaid
graph TD
    subgraph Step 1: Scores
        Q[Q: Query Matrix]
        KT[K^T: Transposed Keys]
        Q --"MatMul (Dot Product)"--> S[Scores]
        KT --> S
    end
    
    subgraph Step 2: Scale & Softmax
        S --"Divide by sqrt(d_k)"--> Sc[Scaled]
        Sc --"Softmax"--> P[Probabilities]
    end
    
    subgraph Step 3: Weighted Sum
        V[V: Values]
        P --"MatMul"--> O[Output]
        V --> O
    end
```
*Diagram: The flow of Scaled Dot-Product Attention. Shapes transform from `[Seq, HeadDim]` to `[Seq, Seq]` (Scores) and back to `[Seq, HeadDim]` (Output).*

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Let's break this down step-by-step:

1.  **\\(QK^T\\)**: We take the dot product of the Query matrix and the Transposed Key matrix. This gives us a **score matrix** of shape `[SequenceLength, SequenceLength]`. Each entry \\(i, j\\) tells us how much the \\(i\\)-th token cares about the \\(j\\)-th token.
2.  **\\(\frac{1}{\sqrt{d_k}}\\\)**: We divide by the square root of the head dimension. This is the "scaling" part. As dimensions grow, dot products can get very large, pushing the softmax into regions where gradients are tiny (vanishing gradients). Scaling keeps the variance stable (similar to Xavier initialization).
3.  **\\(\text{softmax}(\dots)\\\)**: We apply softmax to convert these raw scores into probabilities. For each query, the attention weights across all keys sum to 1.0.
4.  **\\(\dots V\\)**: Finally, we multiply these probabilities by the Value matrix \\(V\\). This computes a weighted sum of the Values.

If the attention weight for "cat" attending to "meow" is 0.9, the output vector will be 90% composed of the "meow" value vector.

## Multi-Head Attention (MHA)

One attention head is good, but it might focus only on one type of relationship (e.g., syntax). We want the model to understand multiple aspects of language simultaneously (syntax, semantics, tone, etc.).

**Multi-Head Attention** runs several attention mechanisms in parallel. Each "head" has its own unique projection matrices (\\(W_Q, W_K, W_V\\)), allowing it to specialize.

- Head 1 might track subject-verb agreement.
- Head 2 might track pronoun antecedents.
- Head 3 might track emotional sentiment.

After computing attention for all heads, we concatenate their outputs and project them back to the model dimension using an output matrix \\(W_O\\).

## Modern Optimizations: Grouped Query Attention (GQA)

In the original Transformer (MHA), every Query head had a corresponding Key and Value head.
- 8 Query Heads \\(\rightarrow\\) 8 Key Heads, 8 Value Heads.

This is great for quality but expensive for memory. During inference (specifically decoding), we must cache all the Keys and Values (the **KV Cache**). A large KV cache consumes massive GPU memory bandwidth.

**Multi-Query Attention (MQA)** proposed an extreme solution:
- 8 Query Heads \\(\rightarrow\\) 1 Key Head, 1 Value Head.
This saves memory but can degrade performance.

**Grouped Query Attention (GQA)** is the "Goldilocks" solution used in modern models like **Gemma** and **Llama 3**. We group multiple query heads to share a single KV head.
- 8 Query Heads \\(\rightarrow\\) 2 Key Heads, 2 Value Heads (4 Queries per KV).

```mermaid
graph TD
    subgraph MHA [Multi-Head Attention]
        direction TB
        Q1(Q1) --> KV1[KV1]
        Q2(Q2) --> KV2[KV2]
        Q3(Q3) --> KV3[KV3]
        Q4(Q4) --> KV4[KV4]
    end

    subgraph GQA [Grouped Query Attention]
        direction TB
        GQ1(Q1) --> GKV1[KV1]
        GQ2(Q2) --> GKV1
        GQ3(Q3) --> GKV2[KV2]
        GQ4(Q4) --> GKV2
    end
```
*Diagram: Comparison of Multi-Head Attention (Left) vs. Grouped Query Attention (Right). In GQA, multiple query heads share a single KV head.*

This significantly reduces the size of the KV cache while maintaining model performance close to MHA.

## The Problem of Order: Rotary Positional Embeddings (RoPE)

The attention mechanism we described (\\(QK^T\\)) is **permutation invariant**. If you shuffle the words in a sentence, the dot products remain the same (just in different positions). "Sam ate an apple" and "An apple ate Sam" would look identical to the attention layer without extra help.

We need to inject **positional information**.

### Absolute vs. Relative

- **Absolute Positional Embeddings** (Original Transformer): Added a fixed vector to each token based on its index (\\(0, 1, 2 \dots\\)).
- **Relative Positional Embeddings**: Learned the distance between tokens.

**RoPE** (Rotary Positional Embeddings) takes a different approach. Instead of adding a vector, it **rotates** the Query and Key vectors in the complex plane.

### The Rotation Intuition

Imagine the embedding vector as a clock hand.
- Token at position 0: No rotation.
- Token at position 1: Rotate by \\(\theta\\).
- Token at position 2: Rotate by \\(2\theta\\).

The angle between two vectors now depends on their **relative distance** (\\(m - n\\)).
$$ (Rm)^T (Rn) = \cos((m-n)\theta) $$

This allows the model to naturally generalize to sequence lengths longer than it saw during training, a significant advantage over the fixed sinusoidal embeddings used in early JAX/PyTorch tutorials.

![RoPE Rotation](images/rope_visualization.png)
*Image Credit: [EleutherAI](https://blog.eleuther.ai/rotary-embeddings/)*

## Implementation in `xla-rs`

Let's look at how this comes together in our codebase. Open `src/models/gemma/attention.rs`.

### The `MultiHeadAttention` Struct

```rust
pub struct MultiHeadAttention<T: TensorElem> {
    pub q_proj: Linear<T>,
    pub k_proj: Linear<T>,
    pub v_proj: Linear<T>,
    pub o_proj: Linear<T>,
    pub num_heads: usize,
    pub num_kv_heads: usize, // For GQA
    pub head_dim: usize,
    pub scaling: T,
}
```

Notice `num_kv_heads`. If `num_kv_heads < num_heads`, we are using GQA.

### The Forward Pass

The `forward` method orchestrates the dance:

1.  **Projection**:
    ```rust
    let q = self.q_proj.forward(x)?;
    let k = self.k_proj.forward(x)?;
    let v = self.v_proj.forward(x)?;
    ```

2.  **Reshape and Transpose**:
    We reshape the linear output to separate heads.
    `[Batch, SeqLen, Hidden]` \\(\rightarrow\\) `[Batch, SeqLen, Heads, HeadDim]`
    Then we transpose to get heads into the leading dimension for parallel processing:
    `[Batch, Heads, SeqLen, HeadDim]`
    ```rust
    let q = q.reshape([b, s, self.num_heads, self.head_dim])?;
    let q = q.transpose_axes(1, 2)?; // Swap SeqLen and Heads
    ```

3.  **Apply RoPE**:
    We apply rotation *before* the dot product.
    ```rust
    let q = apply_rope(&q, freqs_cos, freqs_sin)?;
    let k = apply_rope(&k, freqs_cos, freqs_sin)?;
    ```
    *Note: `apply_rope` in `src/nn/transformer/rope.rs` handles the complex rotation logic.*

4.  **Handle GQA (Repeat KV)**:
    If we have fewer KV heads, we must repeat them to match the number of Query heads so we can perform matrix multiplication.
    ```rust
    let (k, v) = if self.num_kv_heads != self.num_heads {
        (self.repeat_kv(&k)?, self.repeat_kv(&v)?)
    } else {
        (k, v)
    };
    ```

5.  **Attention Scores**:
    $$ Q @ K^T $$
    ```rust
    let k_t = k.transpose()?; // [Batch, Heads, HeadDim, SeqLen]
    let mut scores = q.matmul(&k_t)?;
    scores = scores.map(|val| val * self.scaling);
    ```

6.  **Softmax and Output**:
    ```rust
    self.softmax_inplace(&mut scores)?;
    let output = scores.matmul(&v)?;
    ```

This implementation is efficient, leveraging `xla-rs`'s tensor operations to execute in parallel on the CPU (or GPU/TPU via XLA in the future).
