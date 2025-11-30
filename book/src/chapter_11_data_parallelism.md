# Chapter 11: Data Parallelism & Sharding

> \[!NOTE]
> This chapter is under construction.

## Breaking the Memory Wall

We have split the model within a layer (Tensor Parallelism) and across layers (Pipeline Parallelism). But what if we just want to train *faster*?

**Data Parallelism (DP)** is the standard answer. Replicate the model on $N$ GPUs. Give each GPU a different slice of the data batch. Average the gradients.

But there is a problem. A 175B parameter model (GPT-3) requires 700GB just to store the weights in float32. It doesn't fit on an 80GB A100. You can't replicate it.

Does this mean we can't use Data Parallelism? No. It means we need to be smarter about *what* we replicate.

## The Memory Pie

Where does GPU memory go during training?
1.  **Parameters ($P$)**: The weights.
2.  **Gradients ($P$)**: The calculated updates.
3.  **Optimizer States ($K \cdot P$)**: The history. For Adam, we store Momentum and Variance. That's $2P$.
4.  **Activations**: The intermediate values (depends on batch size).

For mixed-precision training (FP16/BF16), the Optimizer States are often stored in FP32 (Master Weights) to preserve precision. This is huge.
**Total Memory $\approx 20 \times$ Model Size.**

## ZeRO: The Art of Sharding

**ZeRO (Zero Redundancy Optimizer)** asked a simple question: Why are we storing the exact same Optimizer States on every single GPU?

If we have 100 GPUs, we are storing 100 copies of the Adam state. That's 99 copies of waste.

### Stage 1: Shard the Optimizer
Each GPU keeps the full parameters, but is responsible for updating only **1/N** of them.
-   GPU 0 updates weights 0-100.
-   GPU 1 updates weights 101-200.
-   **Memory Savings**: 4x.

### Stage 2: Shard the Gradients
As soon as a gradient is computed, we send it to the GPU responsible for that weight and delete it. We don't hold the full gradient vector.
-   **Memory Savings**: 8x.

### Stage 3: Shard the Parameters
This is the big one. We don't even store the full model weights.
-   GPU 0 holds slice 0 of the weights.
-   GPU 1 holds slice 1.
-   **Computation**: When GPU 0 needs the weights for Layer 5 to do a forward pass, it asks *everyone* for their piece of Layer 5 (`All-Gather`). It computes. Then it **deletes** the weights.
-   **Memory Savings**: Linear with $N$. You can train a trillion-parameter model.

## FSDP (Fully Sharded Data Parallel)

FSDP is PyTorch's implementation of ZeRO-3. It integrates tightly with the training loop.

1.  **Forward Pass**:
    -   `All-Gather` parameters for the current layer.
    -   Compute output.
    -   Discard parameters.
2.  **Backward Pass**:
    -   `All-Gather` parameters.
    -   Compute gradients.
    -   `Reduce-Scatter` gradients (sum them up and scatter them to their owners).
    -   Discard parameters.
3.  **Optimizer Step**:
    -   Update local parameters using local optimizer state.

## The XLA Perspective: GSPMD

In XLA (and JAX), we don't usually think about "ZeRO stages". We think about **Global Arrays**.

**GSPMD (General and Scalable Parallelization for ML Models)** allows us to write code as if we had one giant GPU.
-   We define a **Mesh** of devices (e.g., $4 \times 8$).
-   We annotate our tensors: "This weight matrix is sharded along axis 0 of the mesh."
-   The XLA compiler **automatically inserts the collectives**.

If you multiply a sharded matrix by a replicated matrix, XLA sees the mismatch and inserts an `All-Gather`. This is incredibly powerful because it decouples the *mathematics* of your model from the *physics* of its execution.

### Example: Sharding a Linear Layer

```rust,ignore
// In XLA, we might define a sharding strategy
let mesh = create_mesh(4, 8); // 32 devices
let sharding = HloSharding::tile(mesh, vec![0, 1]); // Split dim 0 on mesh axis 0, dim 1 on mesh axis 1

// Apply to the weight tensor
let w = builder.parameter(..., sharding)?;
```

## The Grand Unification: 3D Parallelism

To train the largest models in the world, we combine everything.

### Example: The 32-GPU Cluster
Let's go back to our 4 nodes x 8 GPUs setup. How do we configure it?

**Scenario A: Llama 70B (Fits on 1 Node)**
-   **Tensor Parallelism = 8**: We split the model across all 8 GPUs in a node.
-   **Pipeline Parallelism = 1**: No pipelining needed.
-   **Data Parallelism = 4**: We replicate this 8-GPU setup 4 times (across the 4 nodes) to increase batch size.

**Scenario B: GPT-4 Scale (Requires multiple nodes)**
-   **Tensor Parallelism = 8**: Maximize intra-node bandwidth.
-   **Pipeline Parallelism = 4**: Split the model depth across the 4 nodes.
-   **Data Parallelism = 1**: We are using the entire cluster to train just one instance of the model.

This is how we reach the Exascale.
