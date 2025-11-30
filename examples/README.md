# xla-rs Examples

This directory contains example binaries demonstrating the capabilities of the `xla-rs` framework. The binaries are located in `src/bin/gemma/`.

> [!IMPORTANT]
> These examples demonstrate a **simplified** Gemma architecture. They do not include advanced features like Interleaved Local-Global Attention, Vision Encoders (PaliGemma), or advanced RL strategies (WARP/BOND).

## Available Binaries

All examples can be run using `cargo run -p xla-rs-examples --bin <binary_name>`.

| Binary Name | Description | Key Concepts | Command |
|-------------|-------------|--------------|---------|
| **`gemma_train`** | **Pre-training Loop** | Basic Next Token Prediction, `Variable` autograd, `Sgd` optimizer. | `cargo run -p xla-rs-examples --bin gemma_train` |
| **`gemma_sft`** | **Supervised Fine-Tuning** | Instruction masking, loss calculation on response tokens only. | `cargo run -p xla-rs-examples --bin gemma_sft` |
| **`gemma_lora`** | **LoRA Fine-Tuning** | Low-Rank Adaptation, freezing base weights, updating adapters ($W + BA$). | `cargo run -p xla-rs-examples --bin gemma_lora` |
| **`gemma_dpo`** | **Alignment (DPO)** | Direct Preference Optimization, Policy vs. Reference model, Preference Loss. | `cargo run -p xla-rs-examples --bin gemma_dpo` |
| **`gemma_serve`** | **Inference Server** | HTTP API using `axum`, Greedy Decoding generation loop. | `cargo run -p xla-rs-examples --bin gemma_serve` |

## Prerequisites

- Rust 1.75+
- No external hardware dependencies (runs on CPU).

## Usage Details

### Training Examples (`train`, `sft`, `lora`, `dpo`)
These examples run a "toy" training loop for 5 epochs on synthetic data. They demonstrate the mechanics of the training process, including:
- Forward pass with `Variable` for automatic differentiation.
- Loss calculation (MSE or DPO).
- Backward pass (`backward()`).
- Optimizer step (`Sgd`).

### Serving Example (`serve`)
The serving example starts a local HTTP server on port 3000.
- **Endpoint**: `POST /generate`
- **Payload**: `{"prompt": "Hello"}`
- **Response**: `{"response": "Hello [Generated Response]"}`

Example request:
```bash
curl -X POST http://127.0.0.1:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is xla-rs?"}'
```
