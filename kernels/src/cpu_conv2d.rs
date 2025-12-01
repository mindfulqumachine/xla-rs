use crate::{KernelElem, KernelError, Result};
use rayon::prelude::*;

/// Performs 2D Convolution on CPU.
///
/// # Arguments
///
/// * `input` - Input tensor data (flattened). Shape: `[batch_size, in_channels, height, width]`
/// * `weight` - Weight tensor data (flattened). Shape: `[out_channels, in_channels, kernel_h, kernel_w]`
/// * `input_shape` - Shape of the input tensor.
/// * `weight_shape` - Shape of the weight tensor.
/// * `stride` - Stride of the convolution: `[stride_h, stride_w]`
/// * `padding` - Padding added to both sides of the input: `[pad_h, pad_w]`
/// * `dilation` - Dilation of the kernel: `[dil_h, dil_w]`
///
/// # Returns
///
/// A flattened vector containing the result of the convolution.
/// Output shape: `[batch_size, out_channels, out_h, out_w]`
pub fn cpu_conv2d<T: KernelElem>(
    input: &[T],
    weight: &[T],
    input_shape: &[usize],
    weight_shape: &[usize],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Result<Vec<T>> {
    // Validate ranks
    if input_shape.len() != 4 {
        return Err(KernelError::ShapeMismatch {
            expected: vec![4], // Rank 4
            got: vec![input_shape.len()],
        });
    }
    if weight_shape.len() != 4 {
        return Err(KernelError::ShapeMismatch {
            expected: vec![4], // Rank 4
            got: vec![weight_shape.len()],
        });
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];

    let out_channels = weight_shape[0];
    let weight_in_channels = weight_shape[1];
    let k_h = weight_shape[2];
    let k_w = weight_shape[3];

    if in_channels != weight_in_channels {
        return Err(KernelError::ShapeMismatch {
            expected: vec![in_channels],
            got: vec![weight_in_channels],
        });
    }

    let stride_h = stride[0];
    let stride_w = stride[1];
    let pad_h = padding[0];
    let pad_w = padding[1];
    let dil_h = dilation[0];
    let dil_w = dilation[1];

    // Compute output dimensions
    // H_out = floor((H + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    let effective_k_h = k_h + (k_h - 1) * (dil_h - 1);
    let effective_k_w = k_w + (k_w - 1) * (dil_w - 1);

    if in_h + 2 * pad_h < effective_k_h {
        return Err(KernelError::ShapeMismatch {
            expected: vec![effective_k_h], // Minimum height
            got: vec![in_h + 2 * pad_h],
        });
    }
    if in_w + 2 * pad_w < effective_k_w {
        return Err(KernelError::ShapeMismatch {
            expected: vec![effective_k_w], // Minimum width
            got: vec![in_w + 2 * pad_w],
        });
    }

    let out_h = (in_h + 2 * pad_h - effective_k_h) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - effective_k_w) / stride_w + 1;

    let out_size = batch_size * out_channels * out_h * out_w;
    let mut output = vec![T::zero(); out_size];

    // Strides for input access
    let in_stride_b = in_channels * in_h * in_w;
    let in_stride_c = in_h * in_w;
    let in_stride_h = in_w;
    let in_stride_w = 1;

    // Strides for weight access
    let w_stride_out = in_channels * k_h * k_w;
    let w_stride_in = k_h * k_w;
    let w_stride_h = k_w;
    let w_stride_w = 1;

    // Strides for output access
    let out_stride_b = out_channels * out_h * out_w;
    let out_stride_c = out_h * out_w;
    let out_stride_h = out_w;
    // let out_stride_w = 1;

    // Parallelize over Batch and Out Channels
    output
        .par_chunks_mut(out_stride_b) // Split by batch
        .enumerate()
        .for_each(|(b, batch_out)| {
            // batch_out is [out_channels, out_h, out_w]
            batch_out
                .par_chunks_mut(out_stride_c) // Split by out_channel
                .enumerate()
                .for_each(|(oc, channel_out)| {
                    // channel_out is [out_h, out_w]
                    // We need to convolve input[b, :, :, :] with weight[oc, :, :, :]

                    // Pre-calculate weight offset for this output channel
                    let w_offset_base = oc * w_stride_out;

                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = T::zero();

                            // Input window start
                            let h_start = (oh * stride_h) as isize - pad_h as isize;
                            let w_start = (ow * stride_w) as isize - pad_w as isize;

                            for ic in 0..in_channels {
                                let in_offset_base = b * in_stride_b + ic * in_stride_c;
                                let w_offset_ic = w_offset_base + ic * w_stride_in;

                                for kh in 0..k_h {
                                    for kw in 0..k_w {
                                        let h_in = h_start + (kh * dil_h) as isize;
                                        let w_in = w_start + (kw * dil_w) as isize;

                                        if h_in >= 0
                                            && h_in < in_h as isize
                                            && w_in >= 0
                                            && w_in < in_w as isize
                                        {
                                            let h_idx = h_in as usize;
                                            let w_idx = w_in as usize;

                                            let in_idx = in_offset_base
                                                + h_idx * in_stride_h
                                                + w_idx * in_stride_w;
                                            let w_idx =
                                                w_offset_ic + kh * w_stride_h + kw * w_stride_w;

                                            let val_in = input[in_idx];
                                            let val_w = weight[w_idx];

                                            // sum += val_in * val_w
                                            // T needs to support Mul and AddAssign
                                            // KernelElem supports NumAssign which implies AddAssign and Mul
                                            let prod = val_in * val_w;
                                            sum += prod;
                                        }
                                    }
                                }
                            }

                            channel_out[oh * out_stride_h + ow] = sum;
                        }
                    }
                });
        });

    Ok(output)
}
