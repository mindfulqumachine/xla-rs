# Optimization & Training Dynamics

> "The difference between a model that learns and one that doesn't is often just the learning rate."

In the previous chapters, we built the architecture of GPT-2. However, defining the model is only half the battle. The other half is **training dynamics**: the art and science of guiding your model's weights to a minimum on the loss landscape without exploding, vanishing, or getting stuck.

In this chapter, we will implement the critical components that stabilize training:
1.  **AdamW Optimizer**: Why SGD isn't enough.
2.  **Learning Rate Schedulers**: The warmup and decay dance.
3.  **Gradient Clipping**: Putting a speed limit on updates.
4.  **Checkpointing**: Saving your progress.

---

## 1. The Optimizer: Beyond SGD

Stochastic Gradient Descent (SGD) is the foundation of deep learning, but for Transformers, it is rarely sufficient. The loss landscape of a Transformer is complex, with many saddle points and varying curvatures.

### Why AdamW?

**Adam** (Adaptive Moment Estimation) adapts the learning rate for *each parameter* individually.
- Parameters with large gradients get smaller updates (preventing oscillation).
- Parameters with small gradients get larger updates (accelerating convergence).

**AdamW** fixes a flaw in the original Adam implementation regarding weight decay. In AdamW, weight decay is decoupled from the gradient update, which leads to better generalization.

### Implementation in `xla-rs`

We've implemented `AdamW` in `core/src/optim/adamw.rs`. Here's how you use it:

```rust
use xla_rs::optim::{AdamW, Optimizer};

// 1. Create the optimizer
let mut optimizer = AdamW::new(3e-4); // Learning rate

// 2. In the training loop:
// Calculate gradients...
let grads = model.backward(&loss);

// Update parameters
optimizer.update(model.parameters(), grads, step)?;
```

---

## 2. Learning Rate Schedulers

If you train a Transformer with a constant learning rate, it will likely diverge or converge to a poor solution. We need a **Schedule**.

### The Warmup
At the beginning of training, the model's weights are random. Large updates can push the weights into bad regions of the loss landscape from which they can't recover.
**Warmup** linearly increases the learning rate from 0 to `max_lr` over the first few thousand steps. This allows the model to stabilize before "real" training begins.

### The Decay
As the model approaches a minimum, we want to take smaller steps to settle into the valley.
**Cosine Decay** is the standard choice. It smoothly decreases the LR from `max_lr` to `min_lr` following a cosine curve.

### Using Schedulers

We provide `LinearWarmup` and `CosineDecay` in `core/src/optim/scheduler.rs`.

```rust
use xla_rs::optim::scheduler::{LRScheduler, LinearWarmup, CosineDecay};

let warmup_steps = 2000;
let total_steps = 100_000;

// Combine schedulers logic in your training loop:
let lr = if step < warmup_steps {
    LinearWarmup::new(0.0, 3e-4, warmup_steps).get_lr(step)
} else {
    CosineDecay::new(3e-4, 3e-5, total_steps - warmup_steps).get_lr(step - warmup_steps)
};

optimizer.set_lr(lr);
```

---

## 3. Gradient Clipping

Transformers can suffer from **exploding gradients**, where a single bad batch produces massive gradients that wreck the model weights.

**Gradient Clipping** scales down the entire gradient vector if its norm exceeds a threshold (usually 1.0). This preserves the *direction* of the update but limits its *magnitude*.

```rust
// In your training loop, before optimizer.update():
let total_norm = xla_rs::optim::utils::clip_grad_norm(&mut grads, 1.0);
```

---

## 4. Checkpointing & Serialization

Training a LLM takes days or weeks. You need to save your progress.
We use `safetensors` for efficient, safe serialization.

### Saving

```rust
use xla_rs::checkpoint::save_checkpoint;

// Save model weights
save_checkpoint("model.safetensors", &model.state_dict())?;

// Save optimizer state (momentum, variance)
save_checkpoint("optimizer.safetensors", &optimizer.state_dict())?;
```

### Loading

```rust
use xla_rs::checkpoint::load_checkpoint;

// Load weights
let weights = load_checkpoint("model.safetensors")?;
model.load_state_dict(&weights)?;

// Resume optimizer
let optim_state = load_checkpoint("optimizer.safetensors")?;
optimizer.load_state_dict(&optim_state)?;
```

---

## Summary

With AdamW, Schedulers, Clipping, and Checkpointing, you now have a robust training loop capable of training serious models. In the next chapter, we will explore **Distributed Training** to scale this up to multiple GPUs.
