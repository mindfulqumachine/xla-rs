use crate::nn::Module;
use crate::tensor::{Device, Result, Tensor, TensorElem};

/// 2D Convolution Layer.
///
/// Applies a 2D convolution over an input signal composed of several input planes.
pub struct Conv2d<T, D: Device>
where
    T: TensorElem,
{
    pub weight: Tensor<T, 4, D>,
    pub bias: Option<Tensor<T, 1, D>>,
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

impl<T> Conv2d<T, crate::tensor::Cpu>
where
    T: TensorElem + num_traits::Float,
{
    /// Creates a new Conv2d layer.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of channels in the input image.
    /// * `out_channels` - Number of channels produced by the convolution.
    /// * `kernel_size` - Size of the convolving kernel.
    /// * `stride` - Stride of the convolution.
    /// * `padding` - Zero-padding added to both sides of the input.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Conv2d<T, crate::tensor::Cpu> {
        // Initialize weights (using simple initialization for now, e.g., Kaiming/Xavier is better)
        // Shape: [out_channels, in_channels, kH, kW]
        let weight = Tensor::zeros([out_channels, in_channels, kernel_size, kernel_size]);
        let bias = Some(Tensor::zeros([out_channels]));

        Conv2d {
            weight,
            bias,
            stride: [stride, stride],
            padding: [padding, padding],
            dilation: [1, 1],
            groups: 1,
        }
    }

    /// Performs the forward pass.
    pub fn forward(
        &self,
        input: &Tensor<T, 4, crate::tensor::Cpu>,
    ) -> Result<Tensor<T, 4, crate::tensor::Cpu>> {
        let out = input.conv2d(&self.weight, self.stride, self.padding, self.dilation)?;

        if let Some(bias) = &self.bias {
            // Bias shape: [out_channels]
            // Output shape: [B, out_channels, H, W]
            // We need to broadcast bias to [1, out_channels, 1, 1] and add.
            // Currently `add` requires exact shape match or simple broadcasting?
            // `ops.rs` says "Currently, xla-rs implements strict shape checking".
            // So we need to reshape bias and broadcast manually or implement broadcasting.
            // Or use a loop like in Linear.

            // Let's use a loop for now to be safe and explicit, similar to Linear.
            let out_shape = out.shape();
            let _batch = out_shape[0];
            let out_c = out_shape[1];
            let h = out_shape[2];
            let w = out_shape[3];

            if bias.shape()[0] != out_c {
                return Err(crate::tensor::TensorError::ShapeMismatch {
                    expected: vec![out_c],
                    got: vec![bias.shape()[0]],
                });
            }

            let mut out_mut = out;
            let out_data = out_mut.data_mut();
            let bias_data = bias.data();

            // Parallelize over batch and channel
            use rayon::prelude::*;
            let stride_b = out_c * h * w;
            let stride_c = h * w;

            out_data.par_chunks_mut(stride_b).for_each(|batch_chunk| {
                batch_chunk
                    .chunks_mut(stride_c)
                    .enumerate()
                    .for_each(|(c, channel_chunk)| {
                        let b_val = bias_data[c];
                        for val in channel_chunk.iter_mut() {
                            *val += b_val;
                        }
                    });
            });

            Ok(out_mut)
        } else {
            Ok(out)
        }
    }
}

// Actually, let's look at `Linear` implementation for reference on how we handle initialization.
