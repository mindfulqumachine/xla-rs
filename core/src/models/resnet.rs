use crate::nn::{Conv2d, LayerNorm, Linear, MaxPool2d};
use crate::tensor::{Cpu, Result, Tensor, TensorElem};

/// A ResNet Basic Block.
///
/// Consists of:
/// - Conv3x3
/// - Norm
/// - ReLU
/// - Conv3x3
/// - Norm
/// - Residual Connection
pub struct BasicBlock<T: TensorElem> {
    pub conv1: Conv2d<T, Cpu>,
    pub norm1: LayerNorm<T>,
    pub conv2: Conv2d<T, Cpu>,
    pub norm2: LayerNorm<T>,
    pub downsample: Option<Conv2d<T, Cpu>>, // Simplified downsample (just conv, no norm usually in simple version or conv+norm)
    // Standard ResNet has norm in downsample too.
    pub downsample_norm: Option<LayerNorm<T>>,
}

impl<T: TensorElem + num_traits::Float> BasicBlock<T> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let conv1 = Conv2d::new(in_channels, out_channels, 3, stride, 1); // 3x3, padding 1
        let norm1 = LayerNorm::new(
            Tensor::ones([out_channels]),
            Tensor::zeros([out_channels]),
            T::from_f32(1e-5).unwrap(),
        );

        let conv2 = Conv2d::new(out_channels, out_channels, 3, 1, 1); // 3x3, stride 1, padding 1
        let norm2 = LayerNorm::new(
            Tensor::ones([out_channels]),
            Tensor::zeros([out_channels]),
            T::from_f32(1e-5).unwrap(),
        );

        let (downsample, downsample_norm) = if stride != 1 || in_channels != out_channels {
            (
                Some(Conv2d::new(in_channels, out_channels, 1, stride, 0)), // 1x1 conv
                Some(LayerNorm::new(
                    Tensor::ones([out_channels]),
                    Tensor::zeros([out_channels]),
                    T::from_f32(1e-5).unwrap(),
                )),
            )
        } else {
            (None, None)
        };

        Self {
            conv1,
            norm1,
            conv2,
            norm2,
            downsample,
            downsample_norm,
        }
    }

    pub fn forward(&self, x: &Tensor<T, 4, Cpu>) -> Result<Tensor<T, 4, Cpu>> {
        let identity = x.clone();

        let out = self.conv1.forward(x)?;
        let out = self.apply_norm(&out, &self.norm1)?;
        let out = out.map(|v| if v > T::zero() { v } else { T::zero() }); // ReLU

        let out = self.conv2.forward(&out)?;
        let out = self.apply_norm(&out, &self.norm2)?;

        let identity = if let Some(ds) = &self.downsample {
            let id = ds.forward(&identity)?;
            if let Some(ds_norm) = &self.downsample_norm {
                self.apply_norm(&id, ds_norm)?
            } else {
                id
            }
        } else {
            identity
        };

        // out + identity
        // Tensor add requires explicit reference?
        // &out + &identity
        let out = (&out + &identity)?;
        let out = out.map(|v| if v > T::zero() { v } else { T::zero() }); // ReLU

        Ok(out)
    }

    /// Helper to apply LayerNorm to [B, C, H, W] tensor.
    /// Permutes to [B, H, W, C], applies norm, permutes back.
    fn apply_norm(&self, x: &Tensor<T, 4, Cpu>, norm: &LayerNorm<T>) -> Result<Tensor<T, 4, Cpu>> {
        // [B, C, H, W] -> [B, H, W, C]
        // Permute: 0, 2, 3, 1
        // My transpose_axes only supports swapping 2 axes.
        // I need general permute or chain swaps.
        // [B, C, H, W] -> swap(1, 2) -> [B, H, C, W] -> swap(2, 3) -> [B, H, W, C]
        let x_perm = x.transpose_axes(1, 2)?.transpose_axes(2, 3)?;

        let x_norm = norm.forward(&x_perm)?;

        // [B, H, W, C] -> swap(2, 3) -> [B, H, C, W] -> swap(1, 2) -> [B, C, H, W]
        let x_out = x_norm.transpose_axes(2, 3)?.transpose_axes(1, 2)?;

        Ok(x_out)
    }
}

/// ResNet-18 Architecture.
pub struct ResNet<T: TensorElem> {
    pub conv1: Conv2d<T, Cpu>,
    pub norm1: LayerNorm<T>,
    pub maxpool: MaxPool2d,

    pub layer1: Vec<BasicBlock<T>>,
    pub layer2: Vec<BasicBlock<T>>,
    pub layer3: Vec<BasicBlock<T>>,
    pub layer4: Vec<BasicBlock<T>>,

    pub fc: Linear<T>,
}

impl<T: TensorElem + num_traits::Float> ResNet<T> {
    pub fn new(num_classes: usize) -> Self {
        // Initial layers
        // Input: [B, 3, 224, 224]
        let conv1 = Conv2d::new(3, 64, 7, 2, 3); // 7x7, stride 2, pad 3
        let norm1 = LayerNorm::new(
            Tensor::ones([64]),
            Tensor::zeros([64]),
            T::from_f32(1e-5).unwrap(),
        );
        let maxpool = MaxPool2d::new(3, 2, 1); // 3x3, stride 2, pad 1

        let layer1 = Self::make_layer(64, 64, 2, 1);
        let layer2 = Self::make_layer(64, 128, 2, 2);
        let layer3 = Self::make_layer(128, 256, 2, 2);
        let layer4 = Self::make_layer(256, 512, 2, 2);

        let fc = Linear::new(
            Tensor::zeros([num_classes, 512]),
            Some(Tensor::zeros([num_classes])),
        );

        Self {
            conv1,
            norm1,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            fc,
        }
    }

    fn make_layer(
        in_channels: usize,
        out_channels: usize,
        blocks: usize,
        stride: usize,
    ) -> Vec<BasicBlock<T>> {
        let mut layers = Vec::new();
        layers.push(BasicBlock::new(in_channels, out_channels, stride));
        for _ in 1..blocks {
            layers.push(BasicBlock::new(out_channels, out_channels, 1));
        }
        layers
    }

    pub fn forward(&self, x: &Tensor<T, 4, Cpu>) -> Result<Tensor<T, 2, Cpu>> {
        // Initial
        let mut out = self.conv1.forward(x)?;
        // Apply norm (permute logic duplicated, should refactor if possible, but for now inline or helper)
        // Let's use a helper method on ResNet or just duplicate logic since BasicBlock has it private.
        // I'll duplicate for now or make BasicBlock's helper public/shared.
        // Actually, I can't easily share without a trait or util.
        // I'll implement apply_norm here too.
        out = self.apply_norm(&out, &self.norm1)?;
        out = out.map(|v| if v > T::zero() { v } else { T::zero() }); // ReLU
        out = self.maxpool.forward(&out)?;

        // Layers
        for block in &self.layer1 {
            out = block.forward(&out)?;
        }
        for block in &self.layer2 {
            out = block.forward(&out)?;
        }
        for block in &self.layer3 {
            out = block.forward(&out)?;
        }
        for block in &self.layer4 {
            out = block.forward(&out)?;
        }

        // Global Avg Pool
        // [B, 512, H, W] -> [B, 512]
        // Mean over H, W
        let shape = out.shape();
        let b = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let area = T::from_usize(h * w).unwrap();

        // Sum over H, W
        // We can reshape to [B, C, H*W] then sum last dim?
        // Or just iterate.
        // Let's use reshape + matmul with ones? Or just manual loop.
        // Manual loop is safest.
        let mut pooled = Tensor::zeros([b, c]);
        let out_data = out.data();
        let pooled_data = pooled.data_mut();

        // Parallelize over B, C
        // Stride for out: C*H*W
        let stride_b = c * h * w;
        let stride_c = h * w;

        // Use rayon
        use rayon::prelude::*;
        pooled_data
            .par_chunks_mut(c)
            .enumerate()
            .for_each(|(batch_idx, batch_out)| {
                let in_offset_base = batch_idx * stride_b;
                for (ch, out_val) in batch_out.iter_mut().enumerate().take(c) {
                    let mut sum = T::zero();
                    let ch_offset = in_offset_base + ch * stride_c;
                    for i in 0..stride_c {
                        sum += out_data[ch_offset + i];
                    }
                    *out_val = sum / area;
                }
            });

        // FC
        self.fc.forward(&pooled)
    }

    fn apply_norm(&self, x: &Tensor<T, 4, Cpu>, norm: &LayerNorm<T>) -> Result<Tensor<T, 4, Cpu>> {
        let x_perm = x.transpose_axes(1, 2)?.transpose_axes(2, 3)?;
        let x_norm = norm.forward(&x_perm)?;
        let x_out = x_norm.transpose_axes(2, 3)?.transpose_axes(1, 2)?;
        Ok(x_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_basic_block() {
        let in_channels = 16;
        let out_channels = 16;
        let stride = 1;
        let block = BasicBlock::<f32>::new(in_channels, out_channels, stride);

        let batch_size = 2;
        let size = 8;
        let x = Tensor::zeros([batch_size, in_channels, size, size]);
        let out = block.forward(&x).unwrap();

        assert_eq!(out.shape(), &[batch_size, out_channels, size, size]);
    }

    #[test]
    fn test_basic_block_downsample() {
        let in_channels = 16;
        let out_channels = 32;
        let stride = 2;
        let block = BasicBlock::<f32>::new(in_channels, out_channels, stride);

        let batch_size = 2;
        let size = 8;
        let x = Tensor::zeros([batch_size, in_channels, size, size]);
        let out = block.forward(&x).unwrap();

        assert_eq!(out.shape(), &[batch_size, out_channels, size / 2, size / 2]);
    }

    #[test]
    fn test_resnet_forward() {
        let num_classes = 10;
        let resnet = ResNet::<f32>::new(num_classes);

        let batch_size = 2;
        // ResNet expects 3 channels, 224x224 usually, but we can test with smaller if logic allows.
        // The initial conv is 7x7 stride 2 pad 3.
        // Let's use 224 to be safe.
        let x = Tensor::zeros([batch_size, 3, 224, 224]);

        // This might be slow for a unit test if not optimized.
        // Let's try it.
        let out = resnet.forward(&x).unwrap();

        assert_eq!(out.shape(), &[batch_size, num_classes]);
    }
}
