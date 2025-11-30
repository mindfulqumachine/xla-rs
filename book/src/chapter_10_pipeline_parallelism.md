# Chapter 10: Pipeline Parallelism (PP)

> \[!NOTE]
> This chapter is under construction.

## The Assembly Line

Tensor Parallelism (TP) is powerful, but it has a limit: **Communication Overhead**. As you split a tensor across more and more GPUs, the cost of `All-Reduce` grows. Eventually, you spend more time talking than computing.

Furthermore, in our **32-GPU cluster (4 nodes x 8 GPUs)**, TP across nodes is slow because inter-node bandwidth (InfiniBand) is much lower than intra-node bandwidth (NVLink).

**Pipeline Parallelism (PP)** solves this by splitting the model *vertically* across the nodes.
-   **Node 0 (GPUs 0-7)**: Layers 1-20
-   **Node 1 (GPUs 8-15)**: Layers 21-40
-   **Node 2 (GPUs 16-23)**: Layers 41-60
-   **Node 3 (GPUs 24-31)**: Layers 61-80

This looks like an assembly line. Data flows from Node 0 to Node 1, and so on.

## The Bubble Problem

The naive implementation of a pipeline is a disaster for efficiency.

1.  GPU 0 processes a batch. GPU 1 is idle.
2.  GPU 0 sends result to GPU 1.
3.  GPU 1 processes. GPU 0 is idle (waiting for the next batch or the backward pass).

At any given moment, **only one GPU is working**. We call the idle time the **Bubble**.

## GPipe: Filling the Bubble

The Google Brain team introduced **GPipe** to fix this. The idea is simple: **Micro-batches**.

Instead of sending one giant batch of 512 samples, we split it into 32 micro-batches of 16 samples.
-   GPU 0 processes MB1, sends to GPU 1.
-   GPU 0 immediately starts processing MB2.
-   GPU 1 receives MB1 and starts processing.

Now, both GPUs are working simultaneously on different parts of the data.

### The Memory Cost
There is a catch. To perform the backward pass for MB1, we need its activations. But we can't do the backward pass until MB1 reaches the end of the pipeline and comes back.
So, GPU 0 must store the activations for *all* 32 micro-batches before it can start backprop. This causes a massive memory spike.

## 1F1B: The Schedule of Champions

**One-Forward-One-Backward (1F1B)** is the standard schedule used today (e.g., in Megatron-LM).

The goal is to clear memory as fast as possible.
-   As soon as a GPU finishes the forward pass for a micro-batch (and has received the gradient from the next stage), it performs the **backward pass** for that micro-batch.
-   This frees up the activation memory immediately.

The schedule looks like a zipper:
`Fwd1 -> Fwd2 -> Bwd1 -> Fwd3 -> Bwd2 ...`

This keeps the pipeline full (small bubble) *and* keeps memory usage constant (independent of the number of micro-batches).

## Distributed State

In Pipeline Parallelism, the model state is distributed.
-   **KV Cache**: In inference, the KV cache for Layers 1-10 lives on GPU 0. The KV cache for Layers 11-20 lives on GPU 1.
-   **P2P Communication**: Unlike TP, which uses collectives (`All-Reduce`), PP uses Point-to-Point communication (`Send` / `Recv`).

### Implementation in XLA

XLA provides `Send` and `Recv` operations.

```rust,ignore
// Pseudocode for Pipeline Stage
fn pipeline_stage(input: XlaOp, stage_id: usize) -> Result<XlaOp> {
    // 1. Receive from previous stage (if not first)
    let x = if stage_id > 0 {
        xla::recv(previous_rank, ...)?
    } else {
        input
    };

    // 2. Compute Layers
    let y = compute_layers(x)?;

    // 3. Send to next stage (if not last)
    if stage_id < num_stages - 1 {
        xla::send(y, next_rank, ...)?;
    }
    
    Ok(y)
}
```

## Interleaved Stages

To reduce the bubble even further, we can assign multiple "virtual stages" to each physical GPU.
-   **GPU 0**: Layers 1-4 AND Layers 17-20.
-   **GPU 1**: Layers 5-8 AND Layers 21-24.

This is called **Interleaved 1F1B**. It allows GPU 0 to start working on the second chunk of layers while GPU 1 is still busy with the first chunk, smoothing out the workload.
