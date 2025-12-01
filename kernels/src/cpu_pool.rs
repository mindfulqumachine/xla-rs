use crate::{KernelElem, KernelError, Result};
use rayon::prelude::*;

/// Performs 2D Max Pooling on CPU.
///
/// # Arguments
///
/// * `input` - Input tensor data (flattened). Shape: `[batch_size, channels, height, width]`
/// * `input_shape` - Shape of the input tensor.
/// * `kernel_size` - Size of the pooling window: `[k_h, k_w]`
/// * `stride` - Stride of the pooling: `[stride_h, stride_w]`
/// * `padding` - Padding added to both sides of the input: `[pad_h, pad_w]`
///
/// # Returns
///
/// A flattened vector containing the result of the max pooling.
/// Output shape: `[batch_size, channels, out_h, out_w]`
pub fn cpu_max_pool2d<T: KernelElem>(
    input: &[T],
    input_shape: &[usize],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<Vec<T>> {
    if input_shape.len() != 4 {
        return Err(KernelError::ShapeMismatch {
            expected: vec![4],
            got: vec![input_shape.len()],
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];

    let k_h = kernel_size[0];
    let k_w = kernel_size[1];
    let stride_h = stride[0];
    let stride_w = stride[1];
    let pad_h = padding[0];
    let pad_w = padding[1];

    if in_h + 2 * pad_h < k_h {
        return Err(KernelError::ShapeMismatch {
            expected: vec![k_h],
            got: vec![in_h + 2 * pad_h],
        });
    }
    if in_w + 2 * pad_w < k_w {
        return Err(KernelError::ShapeMismatch {
            expected: vec![k_w],
            got: vec![in_w + 2 * pad_w],
        });
    }

    let out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

    let out_size = batch_size * channels * out_h * out_w;
    let mut output = vec![T::zero(); out_size]; // Initialize, but will overwrite

    // Strides
    let in_stride_b = channels * in_h * in_w;
    let in_stride_c = in_h * in_w;
    let in_stride_h = in_w;

    let out_stride_b = channels * out_h * out_w;
    let out_stride_c = out_h * out_w;
    let out_stride_h = out_w;

    // Parallelize over Batch and Channel
    output
        .par_chunks_mut(out_stride_b)
        .enumerate()
        .for_each(|(b, batch_out)| {
            batch_out
                .par_chunks_mut(out_stride_c)
                .enumerate()
                .for_each(|(c, channel_out)| {
                    let in_offset_base = b * in_stride_b + c * in_stride_c;

                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let h_start = (oh * stride_h) as isize - pad_h as isize;
                            let w_start = (ow * stride_w) as isize - pad_w as isize;

                            let mut max_val = T::min_value(); // Use min_value for T

                            // If T doesn't implement Bounded, we might have issues.
                            // KernelElem implements Num which implies ... wait Num doesn't imply Bounded.
                            // But KernelElem implements Primitive which usually has min_value.
                            // Or we can initialize with the first valid element.

                            let mut initialized = false;

                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let h_in = h_start + kh as isize;
                                    let w_in = w_start + kw as isize;

                                    if h_in >= 0
                                        && h_in < in_h as isize
                                        && w_in >= 0
                                        && w_in < in_w as isize
                                    {
                                        let idx = in_offset_base
                                            + (h_in as usize) * in_stride_h
                                            + (w_in as usize);
                                        let val = input[idx];

                                        if !initialized {
                                            max_val = val;
                                            initialized = true;
                                        } else if val > max_val {
                                            max_val = val;
                                        }
                                    }
                                }
                            }

                            // If padding makes the window completely out of bounds (shouldn't happen with correct logic),
                            // max_val might be uninitialized. But with pad < kernel_size usually it overlaps something.
                            // If purely padding, max pooling usually returns -inf or padding value.
                            // Here we assume standard valid padding.

                            channel_out[oh * out_stride_h + ow] = max_val;
                        }
                    }
                });
        });

    Ok(output)
}
