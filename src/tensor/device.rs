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
}
