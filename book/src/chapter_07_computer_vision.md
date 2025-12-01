# Computer Vision: Beyond Text

While Large Language Models (LLMs) have taken the world by storm, Computer Vision (CV) remains a critical domain of AI. From self-driving cars to medical imaging, understanding visual data is essential.

In this chapter, we explore how `xla-rs` supports vision architectures, specifically **ResNet** (Convolutional Neural Networks) and **Vision Transformers (ViT)**.

## Convolutions: The ResNet Architecture

Convolutional Neural Networks (CNNs) process images by sliding filters (kernels) over the input. This allows the model to learn spatial hierarchies of features—edges, textures, and eventually complex objects.

### The Convolution Operation

In `xla-rs`, a 2D convolution is implemented via `Conv2d`. It takes an input tensor of shape `[Batch, Channels, Height, Width]` and produces feature maps.

```rust,ignore
use xla_rs::nn::Conv2d;
use xla_rs::tensor::{Tensor, Cpu};

// Input: [Batch=1, Channels=3, Height=32, Width=32]
let x = Tensor::<f32, 4, Cpu>::zeros([1, 3, 32, 32]);

// Conv: 3 input channels -> 16 output channels, 3x3 kernel
let conv = Conv2d::new(3, 16, 3, 1, 1);
let out = conv.forward(&x).unwrap();

assert_eq!(out.shape(), &[1, 16, 32, 32]);
```

### Residual Connections

ResNet introduced the concept of **Residual Connections** (skip connections), allowing gradients to flow more easily during training. A basic ResNet block looks like this:

$$ y = \mathcal{F}(x, \{W_i\}) + x $$

Where $\mathcal{F}$ is the residual function (e.g., two convolution layers).

## Vision Transformers (ViT): Patching the World

Transformers, originally designed for text, can also process images. The key idea behind the **Vision Transformer (ViT)** is to treat an image as a sequence of "patches".

### Patch Embeddings

Instead of processing pixels individually, we divide the image into fixed-size patches (e.g., 16x16). Each patch is flattened and projected into a vector embedding.

In `xla-rs`, we implement this efficiently using a Convolution with `kernel_size` and `stride` equal to the patch size.

```rust,ignore
use xla_rs::nn::Conv2d;

pub struct PatchEmbedding<T: TensorElem> {
    pub proj: Conv2d<T, Cpu>,
}

impl<T: TensorElem + num_traits::Float> PatchEmbedding<T> {
    pub fn new(patch_size: usize, in_channels: usize, embed_dim: usize) -> Self {
        // Kernel = Stride = Patch Size
        let proj = Conv2d::new(in_channels, embed_dim, patch_size, patch_size, 0);
        Self { proj }
    }
}
```

### The ViT Architecture

Once patched, the image is just a sequence of vectors, exactly like tokens in an LLM. We add a **Class Token** (CLS) and **Positional Embeddings**, then feed it into a standard Transformer Encoder.

1.  **Patch & Flatten**: Image $\to$ Sequence of Patches.
2.  **Linear Projection**: Map patches to `embed_dim`.
3.  **Add CLS Token**: A learnable token for classification.
4.  **Add Position Embeddings**: Learnable vectors added to each patch to retain spatial information.
5.  **Transformer Encoder**: Self-attention layers.
6.  **MLP Head**: Classification from the CLS token output.

## Summary

`xla-rs` provides the building blocks for both classic CNNs and modern Vision Transformers. By leveraging the same `Tensor` and `Autograd` engine, we can seamlessly mix and match these architectures or even build multi-modal models that process both text and images.
