//! Device abstraction for Tensor storage.
//!
//! # What is a Device?
//!
//! In Deep Learning, a "Device" refers to the hardware component where the tensor data lives and
//! where computations are executed.
//!
//! - **CPU (Central Processing Unit)**: The "brain" of your computer. Good for complex logic,
//!   control flow, and smaller models. `xla-rs` uses the CPU by default.
//! - **GPU (Graphics Processing Unit)**: Specialized hardware with thousands of cores. Essential
//!   for training large models due to massive parallelism.
//! - **TPU (Tensor Processing Unit)**: Google's custom hardware optimized specifically for matrix math.
//!
//! # The `Device` Trait
//!
//! This trait allows `xla-rs` to write code that is "device agnostic". You can define a model once,
//! and run it on CPU, GPU, or even at compile-time (`ConstDevice`) just by changing the generic parameter.

use crate::tensor::{Storage, TensorElem};
use std::fmt::Debug;

/// A trait representing the underlying storage device for a Tensor.
///
/// # Design
///
/// Devices must define:
/// 1. **Storage**: What container holds the data (e.g., `Vec<T>` for CPU).
/// 2. **Operations**: How to perform basic ops like `transpose` (which might involve moving memory).
pub trait Device: Clone + Debug + PartialEq + Send + Sync {
    /// The type of storage used by this device.
    type Storage<T>: Storage<T>
    where
        T: TensorElem;

    /// Returns the name of the device.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use xla_rs::tensor::{Cpu, Device};
    /// let device = Cpu;
    /// assert_eq!(device.name(), "CPU");
    /// ```
    fn name(&self) -> &'static str;

    /// Transposes the data.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to transpose.
    /// * `shape` - The shape of the tensor.
    fn transpose<T: TensorElem, const RANK: usize>(
        data: &Self::Storage<T>,
        shape: &[usize; RANK],
    ) -> crate::tensor::Result<Self::Storage<T>>;
}

/// A CPU Device.
///
/// This is the default device. Data is stored in standard system RAM (`Vec<T>`).
///
/// # Performance Note
/// While CPUs are slower than GPUs for massive matrix multiplications, `xla-rs` uses `rayon`
/// to utilize all CPU cores. This makes it surprisingly fast for inference on "small" LLMs (like Gemma 2B)
/// on modern laptops.
#[derive(Clone, Debug, PartialEq)]
pub struct Cpu;

impl Device for Cpu {
    type Storage<T>
        = Vec<T>
    where
        T: TensorElem;

    fn name(&self) -> &'static str {
        "CPU"
    }

    fn transpose<T: TensorElem, const RANK: usize>(
        data: &Self::Storage<T>,
        shape: &[usize; RANK],
    ) -> crate::tensor::Result<Self::Storage<T>> {
        if RANK < 2 {
            return Err(crate::tensor::TensorError::Unsupported(
                "Transpose requires rank >= 2".into(),
            ));
        }
        xla_rs_kernels::cpu_transpose(data, shape).map_err(|e| match e {
            xla_rs_kernels::KernelError::ShapeMismatch { expected, got } => {
                crate::tensor::TensorError::ShapeMismatch { expected, got }
            }
        })
    }
}

/// A Device for compile-time constants.
///
/// # What is this?
/// This is a unique feature of `xla-rs`. It allows you to define tensors that exist entirely
/// at compile-time.
///
/// # Use Cases
/// - **Pre-computed Tables**: Sine/Cosine tables for RoPE (Rotary Positional Embeddings).
/// - **Fixed Weights**: Small filters or masks that never change.
///
/// Operations on `ConstDevice` tensors are evaluated by the Rust compiler (const eval), meaning
/// they have **zero runtime cost**.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstDevice<const N: usize>;

impl<const N: usize> Device for ConstDevice<N> {
    type Storage<T>
        = [T; N]
    where
        T: TensorElem;

    fn name(&self) -> &'static str {
        "ConstDevice"
    }

    fn transpose<T: TensorElem, const RANK: usize>(
        data: &Self::Storage<T>,
        shape: &[usize; RANK],
    ) -> crate::tensor::Result<Self::Storage<T>> {
        if RANK < 2 {
            return Err(crate::tensor::TensorError::Unsupported(
                "Transpose requires rank >= 2".into(),
            ));
        }

        let mut new_shape = *shape;
        new_shape.swap(RANK - 1, RANK - 2);
        let new_strides = crate::tensor::compute_strides(&new_shape);
        let strides = crate::tensor::compute_strides(shape);

        let mut new_data = [T::zero(); N];

        let mut i = 0;
        while i < N {
            let mut coords = [0; RANK];
            let mut rem = i;
            let mut d = 0;
            while d < RANK {
                coords[d] = rem / strides[d];
                rem %= strides[d];
                d += 1;
            }

            coords.swap(RANK - 1, RANK - 2);

            let mut j = 0;
            let mut k = 0;
            while k < RANK {
                j += coords[k] * new_strides[k];
                k += 1;
            }

            new_data[j] = data[i];
            i += 1;
        }
        Ok(new_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device_name() {
        let device = Cpu;
        assert_eq!(device.name(), "CPU");
    }

    #[test]
    fn test_cpu_device_traits() {
        let device = Cpu;
        let device_clone = device.clone();
        assert_eq!(device, device_clone);
        assert_eq!(format!("{:?}", device), "Cpu");
    }

    #[test]
    fn test_const_device_name() {
        let device = ConstDevice::<4>;
        assert_eq!(device.name(), "ConstDevice");
    }

    #[test]
    fn test_cpu_transpose_error() {
        let data = vec![1.0];
        let shape = [1];
        let result = Cpu::transpose(&data, &shape);
        assert!(matches!(
            result,
            Err(crate::tensor::TensorError::Unsupported(_))
        ));
    }

    #[test]
    fn test_const_device_transpose_error() {
        let data = [1.0];
        let shape = [1];
        let result = ConstDevice::<1>::transpose(&data, &shape);
        assert!(matches!(
            result,
            Err(crate::tensor::TensorError::Unsupported(_))
        ));
    }

    #[test]
    fn test_const_device_transpose_success() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2, 3];
        let result = ConstDevice::<6>::transpose(&data, &shape).unwrap();
        // [1 2 3]
        // [4 5 6]
        // ->
        // [1 4]
        // [2 5]
        // [3 6]
        assert_eq!(result, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_cpu_transpose_mismatch() {
        let data = vec![1.0, 2.0];
        let shape = [2, 2]; // Size 4, data 2 -> Mismatch
        let result = Cpu::transpose(&data, &shape);
        assert!(matches!(
            result,
            Err(crate::tensor::TensorError::ShapeMismatch { .. })
        ));
    }
}
