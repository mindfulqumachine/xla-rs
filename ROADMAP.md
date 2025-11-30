# xla-rs Roadmap

This document outlines the path to making `xla-rs` an excellent educational and production-capable framework.
The development strategy interleaves **Library Features** (coding) with **Book Chapters** (writing) to ensure that every new concept taught in the book is backed by a working implementation.

## Phase 1: The "Zero to Hero" Bridge
*Goal: Enable users to train a GPT-2 model from scratch on their local machine.*

1.  - [x] **[Library] Advanced Optimizers (AdamW)**
    *   *Why:* SGD is insufficient for Transformer training. We need AdamW.
2.  - [x] **[Library] Data Loading Infrastructure**
    *   *Why:* We need to efficiently batch and shuffle data, not just load entire files into RAM.
3.  - [x] **[Book] Chapter: Building GPT-2 from Scratch**
    *   *Why:* Connects the dots between Tensors/Autograd and the Gemma chapter.
    *   *Dependency:* Requires AdamW and DataLoader.

## Phase 2: Training Dynamics & Stability
*Goal: Teach users how to stabilize training and manage experiments.*

4.  - [ ] **[Library] Learning Rate Schedulers**
    *   *Why:* Transformers require warmup and decay to converge.
5.  - [ ] **[Library] Checkpointing & Serialization**
    *   *Why:* Training takes hours; we need to save/resume.
6.  - [ ] **[Book] Chapter: Optimization & Training Dynamics**
    *   *Why:* Explains the "black magic" of training (schedulers, clipping, regularization).
    *   *Dependency:* Requires Schedulers and Checkpointing.

## Phase 3: Beyond Text (Vision)
*Goal: Demonstrate XLA's versatility beyond LLMs.*

7.  - [ ] **[Library] Vision Models (ResNet/ViT)**
    *   *Why:* Computer Vision is a huge domain.
8.  - [ ] **[Book] Chapter: Computer Vision**
    *   *Why:* Shows how Convolutions and Patching work in `xla-rs`.

## Phase 4: Advanced Distributed Training
*Goal: Scale to clusters.*

9.  - [ ] **[Book] Chapter: Advanced Parallelism**
    *   *Why:* Deep dive into FSDP and Pipeline Parallelism.
