use crate::nn::transformer::attention::MultiHeadAttention;
use crate::nn::{Activation, Conv2d, LayerNorm, Linear};
use crate::tensor::{Cpu, Result, Tensor, TensorElem};

/// Patch Embedding Layer.
///
/// Converts a 2D image into a sequence of flattened patches.
/// Implemented as a Convolution with kernel_size = stride = patch_size.
pub struct PatchEmbedding<T: TensorElem> {
    pub proj: Conv2d<T, Cpu>,
    pub num_patches: usize,
}

impl<T: TensorElem + num_traits::Float> PatchEmbedding<T> {
    pub fn new(img_size: usize, patch_size: usize, in_channels: usize, embed_dim: usize) -> Self {
        let proj = Conv2d::new(in_channels, embed_dim, patch_size, patch_size, 0);
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        Self { proj, num_patches }
    }

    pub fn forward(&self, x: &Tensor<T, 4, Cpu>) -> Result<Tensor<T, 3, Cpu>> {
        // x: [B, C, H, W]
        let x = self.proj.forward(x)?; // [B, EmbedDim, H', W']

        // Flatten: [B, EmbedDim, H'*W']
        let shape = x.shape();
        let b = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];

        // We want [B, NumPatches, EmbedDim] for Transformer
        // Current: [B, C, H, W]
        // Flatten H, W -> [B, C, N]
        let x_flat = x.reshape([b, c, h * w])?;

        // Transpose to [B, N, C]
        let x_out = x_flat.transpose_axes(1, 2)?;

        Ok(x_out)
    }
}

/// ViT Transformer Block.
pub struct ViTBlock<T: TensorElem> {
    pub ln1: LayerNorm<T>,
    pub attn: MultiHeadAttention<T>,
    pub ln2: LayerNorm<T>,
    pub mlp_fc1: Linear<T>,
    pub mlp_fc2: Linear<T>,
}

impl<T: TensorElem + num_traits::Float> ViTBlock<T> {
    pub fn new(embed_dim: usize, num_heads: usize, mlp_ratio: usize) -> Self {
        let ln1 = LayerNorm::new(
            Tensor::ones([embed_dim]),
            Tensor::zeros([embed_dim]),
            T::from_f32(1e-6).unwrap(),
        );

        let head_dim = embed_dim / num_heads;
        // Projections for Attention
        let q_proj = Linear::new(
            Tensor::zeros([embed_dim, embed_dim]),
            Some(Tensor::zeros([embed_dim])),
        );
        let k_proj = Linear::new(
            Tensor::zeros([embed_dim, embed_dim]),
            Some(Tensor::zeros([embed_dim])),
        );
        let v_proj = Linear::new(
            Tensor::zeros([embed_dim, embed_dim]),
            Some(Tensor::zeros([embed_dim])),
        );
        let o_proj = Linear::new(
            Tensor::zeros([embed_dim, embed_dim]),
            Some(Tensor::zeros([embed_dim])),
        );

        let attn = MultiHeadAttention::new(
            embed_dim, num_heads, num_heads, head_dim, q_proj, k_proj, v_proj, o_proj,
        );

        let ln2 = LayerNorm::new(
            Tensor::ones([embed_dim]),
            Tensor::zeros([embed_dim]),
            T::from_f32(1e-6).unwrap(),
        );

        let hidden_dim = embed_dim * mlp_ratio;
        let mlp_fc1 = Linear::new(
            Tensor::zeros([hidden_dim, embed_dim]),
            Some(Tensor::zeros([hidden_dim])),
        );
        let mlp_fc2 = Linear::new(
            Tensor::zeros([embed_dim, hidden_dim]),
            Some(Tensor::zeros([embed_dim])),
        );

        Self {
            ln1,
            attn,
            ln2,
            mlp_fc1,
            mlp_fc2,
        }
    }

    pub fn forward(&self, x: &Tensor<T, 3, Cpu>) -> Result<Tensor<T, 3, Cpu>> {
        // Pre-Norm
        let norm1 = self.ln1.forward(x)?;
        let attn_out = self.attn.forward(&norm1, None, None, None)?;
        let x = (x + &attn_out)?; // Residual

        let norm2 = self.ln2.forward(&x)?;
        let mlp_out = self.mlp_fc1.forward(&norm2)?;
        let mlp_out = Activation::gelu(&mlp_out);
        let mlp_out = self.mlp_fc2.forward(&mlp_out)?;

        let x = (&x + &mlp_out)?; // Residual
        Ok(x)
    }
}

/// Vision Transformer (ViT).
pub struct ViT<T: TensorElem> {
    pub patch_embed: PatchEmbedding<T>,
    pub cls_token: Tensor<T, 3, Cpu>, // [1, 1, D]
    pub pos_embed: Tensor<T, 3, Cpu>, // [1, N+1, D]
    pub blocks: Vec<ViTBlock<T>>,
    pub norm: LayerNorm<T>,
    pub head: Linear<T>,
}

impl<T: TensorElem + num_traits::Float> ViT<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_channels: usize,
        num_classes: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        mlp_ratio: usize,
    ) -> Self {
        let patch_embed = PatchEmbedding::new(img_size, patch_size, in_channels, embed_dim);
        let num_patches = patch_embed.num_patches;

        let cls_token = Tensor::zeros([1, 1, embed_dim]);
        let pos_embed = Tensor::zeros([1, num_patches + 1, embed_dim]);

        let mut blocks = Vec::new();
        for _ in 0..depth {
            blocks.push(ViTBlock::new(embed_dim, num_heads, mlp_ratio));
        }

        let norm = LayerNorm::new(
            Tensor::ones([embed_dim]),
            Tensor::zeros([embed_dim]),
            T::from_f32(1e-6).unwrap(),
        );

        let head = Linear::new(
            Tensor::zeros([num_classes, embed_dim]),
            Some(Tensor::zeros([num_classes])),
        );

        Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
            head,
        }
    }

    pub fn vit_base() -> Self {
        Self::new(224, 16, 3, 1000, 768, 12, 12, 4)
    }

    pub fn forward(&self, x: &Tensor<T, 4, Cpu>) -> Result<Tensor<T, 2, Cpu>> {
        // Patch Embed: [B, N, D]
        let x_embed = self.patch_embed.forward(x)?;
        let b = x_embed.shape()[0];

        // Expand CLS token to batch size
        // [1, 1, D] -> [B, 1, D]
        // We can use repeat/broadcast.
        // Let's manually repeat for now.
        let cls_data = self.cls_token.data();
        let embed_dim = self.cls_token.shape()[2];
        let mut cls_batch_data = Vec::with_capacity(b * embed_dim);
        for _ in 0..b {
            cls_batch_data.extend_from_slice(cls_data);
        }
        let cls_tokens = Tensor::new(cls_batch_data, [b, 1, embed_dim])?;

        // Concatenate CLS + Patches -> [B, N+1, D]
        // Need `cat` op.
        // I'll implement a simple cat helper or loop.
        // Concatenate along dim 1.
        let n = x_embed.shape()[1];
        let total_len = 1 + n;
        let mut cat_data = Vec::with_capacity(b * total_len * embed_dim);

        let cls_ptr = cls_tokens.data();
        let x_ptr = x_embed.data();

        for i in 0..b {
            // Append CLS
            let cls_start = i * embed_dim;
            cat_data.extend_from_slice(&cls_ptr[cls_start..cls_start + embed_dim]);
            // Append Patches
            let x_start = i * n * embed_dim;
            cat_data.extend_from_slice(&x_ptr[x_start..x_start + n * embed_dim]);
        }

        let x_cat = Tensor::new(cat_data, [b, total_len, embed_dim])?;

        // Add Pos Embed
        // pos_embed is [1, N+1, D]. Broadcast to [B, N+1, D].
        // Simple add should broadcast if implemented, but `ops.rs` says strict shape checking.
        // So we need to broadcast pos_embed manually or loop.
        // Let's loop.
        let mut x_pos = x_cat;
        let pos_data = self.pos_embed.data();
        let x_pos_data = x_pos.data_mut();

        // Parallelize
        use rayon::prelude::*;
        let stride_b = total_len * embed_dim;
        x_pos_data.par_chunks_mut(stride_b).for_each(|batch_chunk| {
            for (i, val) in batch_chunk.iter_mut().enumerate() {
                *val += pos_data[i];
            }
        });

        // Transformer Blocks
        let mut x = x_pos;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Norm
        x = self.norm.forward(&x)?;

        // Extract CLS token output (index 0)
        // [B, N+1, D] -> [B, D]
        let mut cls_out_data = Vec::with_capacity(b * embed_dim);
        let x_data = x.data();
        for i in 0..b {
            let start = i * stride_b;
            cls_out_data.extend_from_slice(&x_data[start..start + embed_dim]);
        }
        let cls_out = Tensor::new(cls_out_data, [b, embed_dim])?;

        // Head
        self.head.forward(&cls_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_patch_embedding() {
        let img_size = 32;
        let patch_size = 4;
        let in_channels = 3;
        let embed_dim = 16;
        let pe = PatchEmbedding::<f32>::new(img_size, patch_size, in_channels, embed_dim);

        let batch_size = 2;
        let x = Tensor::zeros([batch_size, in_channels, img_size, img_size]);
        let out = pe.forward(&x).unwrap();

        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        assert_eq!(out.shape(), &[batch_size, num_patches, embed_dim]);
    }

    #[test]
    fn test_vit_block() {
        let embed_dim = 32;
        let num_heads = 4;
        let mlp_ratio = 2;
        let block = ViTBlock::<f32>::new(embed_dim, num_heads, mlp_ratio);

        let batch_size = 2;
        let seq_len = 10;
        let x = Tensor::zeros([batch_size, seq_len, embed_dim]);
        let out = block.forward(&x).unwrap();

        assert_eq!(out.shape(), &[batch_size, seq_len, embed_dim]);
    }

    #[test]
    fn test_vit_forward() {
        let img_size = 32;
        let patch_size = 4;
        let in_channels = 3;
        let num_classes = 10;
        let embed_dim = 32;
        let depth = 2;
        let num_heads = 4;
        let mlp_ratio = 2;

        let vit = ViT::<f32>::new(
            img_size,
            patch_size,
            in_channels,
            num_classes,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
        );

        let batch_size = 2;
        let x = Tensor::zeros([batch_size, in_channels, img_size, img_size]);
        let out = vit.forward(&x).unwrap();

        assert_eq!(out.shape(), &[batch_size, num_classes]);
    }

    #[test]
    fn test_vit_base_init() {
        let vit = ViT::<f32>::vit_base();
        assert_eq!(vit.blocks.len(), 12);
    }
}
