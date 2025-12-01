# Advanced Parallelism: FSDP & Beyond

As models grow to billions of parameters, standard Data Parallelism (DP) becomes insufficient because the model weights and optimizer states no longer fit in a single device's memory. This is where **Advanced Parallelism** techniques like **Fully Sharded Data Parallelism (FSDP)** come into play.

## The Memory Bottleneck

In standard DP, every device holds a full copy of the model.
- **Pros**: Simple communication (AllReduce gradients).
- **Cons**: Memory usage scales with model size, not cluster size.

If a model requires 80GB of memory and you have 8 GPUs with 80GB each, standard DP still limits you to an 80GB model.

## Fully Sharded Data Parallelism (FSDP)

FSDP (inspired by ZeRO-3) solves this by **sharding** the model states (parameters, gradients, optimizer states) across all data parallel workers.

### How it Works

1.  **Shard Everything**: Each device only stores a fraction ($1/N$) of the model.
2.  **AllGather on Demand**: When a layer is needed for the forward or backward pass, devices perform an `AllGather` to reconstruct the full weights of *just that layer*.
3.  **Compute**: Perform the forward/backward pass.
4.  **Discard**: Immediately discard the full weights to free up memory, keeping only the local shard.

This allows you to train models that are effectively $N$ times larger than a single GPU's memory.

### FSDP in `xla-rs`

`xla-rs` leverages the XLA compiler's automatic sharding capabilities (GSPMD) to implement FSDP. By annotating tensors with sharding specifications, the compiler handles the complex communications.

```rust,ignore
// Conceptual FSDP in xla-rs
// We define a mesh of devices (e.g., 8 GPUs)
let mesh = Mesh::new(&[0, 1, 2, 3, 4, 5, 6, 7], "data");

// We shard the weights along the "data" axis
let sharding = Sharding::tile_1d(&mesh, 0); // Shard dim 0
let weights = Tensor::randn(...).shard(&sharding);

// The XLA compiler automatically inserts AllGather/ReduceScatter
```

## Pipeline Parallelism

While FSDP shards data/weights, **Pipeline Parallelism** (PP) shards layers. Device 1 holds layers 1-10, Device 2 holds 11-20, etc.

- **Micro-batches**: To keep all devices busy, the batch is split into micro-batches.
- **Bubble**: There is still some idle time (pipeline bubble) at the start and end of a step.

`xla-rs` supports PP via manual device placement of layers.

## Summary

- **Data Parallelism**: Replicate model, split data. Good for small models.
- **Tensor Parallelism**: Split individual tensors. Good for huge layers.
- **Pipeline Parallelism**: Split layers across devices. Good for deep models.
- **FSDP**: Shard model states across data workers. Best for massive models on standard clusters.

By combining these techniques (3D Parallelism), `xla-rs` can scale to train the largest models in existence.
