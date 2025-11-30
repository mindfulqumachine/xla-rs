//! Device abstraction for Tensor storage.
//!
//! This module defines the `Device` trait and the `Cpu` device implementation.
//! Devices determine where tensor data is allocated and how operations are executed.
//!
//! # ML Context
//!
//! In machine learning frameworks, a "Device" represents the hardware accelerator where
//! computation happens. Common devices include:
//! - **CPU**: Central Processing Unit (host memory). Good for sequential logic and small models.
//! - **GPU**: Graphics Processing Unit (device memory). Excellent for massive parallel matrix operations.
//! - **TPU**: Tensor Processing Unit. Specialized hardware for ML workloads.
//!
//! Abstracting the device allows the same neural network code to run on a laptop (CPU)
//! or a cluster of GPUs without modification.

use crate::tensor::{Storage, TensorElem};
use std::fmt::Debug;

/// A trait representing the underlying storage device for a Tensor.
///
/// Devices determine where the data is stored (e.g., CPU, GPU) and how it is accessed.
/// Currently, only `Cpu` is implemented.
///
/// # Design
///
/// This trait is designed to be extensible. Future implementations could include `Cuda` or `Mps`
/// devices. The `Storage` associated type allows each device to define its own memory container
/// (e.g., `Vec<T>` for CPU, `CudaBuffer<T>` for GPU).
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
/// Represents the standard system CPU. Data is stored in system RAM using `Vec<T>`.
/// This is the default device for all tensors.
///
/// # Performance
///
/// Operations on the CPU are generally slower than GPU for large matrix multiplications,
/// but `xla-rs` uses `rayon` to parallelize operations across all available CPU cores,
/// providing decent performance for development and inference on smaller models.
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
/// Stores data in a fixed-size array `[T; N]`.
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

    #[test]
    fn test_const_device_debug() {
        let device = ConstDevice::<4>;
        assert_eq!(format!("{:?}", device), "ConstDevice");
    }
}
