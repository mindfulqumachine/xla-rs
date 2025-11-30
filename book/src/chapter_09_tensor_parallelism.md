# Chapter 9: Tensor Parallelism (TP)

> \[!NOTE]
> This chapter is under construction.

## Splitting the Atom (Matrix)

We have reached the limits of a single GPU. The weights of our model are simply too large to fit in VRAM.

In **Data Parallelism**, we replicate the entire model on every GPU. This is great for throughput, but it doesn't help with memory capacity. If the model doesn't fit on one GPU, it doesn't fit on $N$ GPUs.

**Tensor Parallelism (TP)** is the solution. We **split the weights themselves** across multiple devices. It is "intra-layer" parallelism.

### The Challenge
Consider a simple linear layer: $Y = X \cdot W$.
-   $X$ is the input vector of size $\[1, D_{in}\]$.
-   $W$ is the weight matrix of size $\[D_{in}, D_{out}\]$.
-   $Y$ is the output vector of size $\[1, D_{out}\]$.

If $W$ is too big, we must slice it. But matrix multiplication isn't just a bag of independent operations. Every output element depends on a dot product of an entire row of $X$ and an entire column of $W$.

## The Megatron-LM Insight

The researchers at NVIDIA (Megatron-LM) formalized how to split Transformer blocks efficiently. The key is to use two types of parallelism that cancel out each other's communication requirements.

### 1. Column Parallelism
We split the weight matrix $W$ along its columns (the output dimension).

-   **Split**: $W = \[W_1 | W_2\]$.
-   **Computation**:
    -   GPU 1 computes $Y_1 = X \cdot W_1$.
    -   GPU 2 computes $Y_2 = X \cdot W_2$.
-   **Result**: The output $Y$ is split: $Y = \[Y_1 | Y_2\]$.
-   **Communication**: **None!** If $X$ is replicated (available on both GPUs), each GPU can compute its part of the output independently.

### 2. Row Parallelism
We split the weight matrix $W$ along its rows (the input dimension).

-   **Split**: $W = \begin{bmatrix} W_1 \\ W_2 \end{bmatrix}$.
-   **Input Requirement**: For the math to work, the input $X$ must also be split along its columns: $X = \[X_1 | X_2\]$.
-   **Computation**:
    -   GPU 1 computes $Z_1 = X_1 \cdot W_1$.
    -   GPU 2 computes $Z_2 = X_2 \cdot W_2$.
-   **Result**: The true output $Y$ is the sum of these partial results: $Y = Z_1 + Z_2$.
-   **Communication**: We need an **All-Reduce (Sum)** to get the final $Y$.

### The MLP Sandwich
The genius of Megatron-LM is combining these two to minimize communication. A Transformer MLP block consists of two linear layers with an activation in between.

$$ \text{MLP}(X) = \text{Linear}_2(\sigma(\text{Linear}_1(X))) $$

1.  **Linear 1 (Column Parallel)**:
    -   Input $X$ is replicated.
    -   Weight $A$ is split by column.
    -   Output $Y$ is split by column. **(0 Comm)**
2.  **Activation ($\sigma$)**:
    -   Applied element-wise on the split $Y$. **(0 Comm)**
3.  **Linear 2 (Row Parallel)**:
    -   Input $Y$ is already split by column (which matches Row Parallel's input requirement!).
    -   Weight $B$ is split by row.
    -   Output $Z$ is a partial sum. **(0 Comm)**
4.  **All-Reduce**:
    -   Sum $Z$ across GPUs to get the final output. **(1 Comm)**

We processed an entire MLP block with **only one synchronization point**.

```mermaid
graph TD
    subgraph GPU 1
    X1[Input X (Copy)] --> CP1[Column Linear A1]
    CP1 --> Act1[GELU]
    Act1 --> RP1[Row Linear B1]
    RP1 --> AllReduce
    end
    subgraph GPU 2
    X2[Input X (Copy)] --> CP2[Column Linear A2]
    CP2 --> Act2[GELU]
    Act2 --> RP2[Row Linear B2]
    RP2 --> AllReduce
    end
    AllReduce --> Out[Output Z]
```

## Attention Heads: The Natural Split

Multi-Head Attention (MHA) is even easier. The "heads" are already independent subspaces.

-   **Q, K, V Projections**: These are Column Parallel.
    -   If we have 16 heads and 2 GPUs, GPU 1 gets Heads 1-8, GPU 2 gets Heads 9-16.
-   **Attention Mechanism**: Computed locally. GPU 1 attends only within its own heads.
-   **Output Projection**: This is Row Parallel. It takes the concatenated output of the heads and projects it back.

Again, we only need **one All-Reduce** at the very end of the Attention block.

## Implementation in Rust/XLA

To implement this, we need to tell XLA how our tensors are sharded.

### Sharding Annotations
XLA uses `HloSharding` to describe the layout of a tensor.

-   **Replicated**: The full tensor exists on every device.
-   **Tiled (Sharded)**: The tensor is split along specific dimensions.

```rust,ignore
// Pseudocode for a TP Linear Layer
struct LinearTP {
    weight: XlaOp, // Sharded
    bias: Option<XlaOp>, // Sharded or Replicated
    strategy: ParallelStrategy, // Column or Row
}

impl LinearTP {
    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let out = x.dot(&self.weight)?;
        
        match self.strategy {
            ParallelStrategy::Column => {
                // Output is split. No comm needed yet.
                Ok(out)
            }
            ParallelStrategy::Row => {
                // Output is partial sum. Need All-Reduce.
                let builder = x.builder();
                let reduced = builder.all_reduce(
                    &out,
                    &xla::XlaComputation::add(&builder, ...),
                    &[], 
                    None, 
                    None
                )?;
                Ok(reduced)
            }
        }
    }
}
```

### Randomness
A subtle trap: **Dropout**.
-   If a layer is replicated, we want the *same* dropout mask on all GPUs. We must seed the RNG identically.
-   If a layer is sharded, we might want *different* dropout masks (to drop different neurons).

## Beyond 1D
What we described is **1D Tensor Parallelism**. We split the tensor along one axis.
-   **2D/2.5D/3D Parallelism**: These advanced techniques split the input, weight, and output tensors along multiple dimensions (rows *and* columns) to arrange GPUs in a grid or cube. This reduces the memory redundancy of activations and can lower communication latency for massive clusters (thousands of GPUs).
