//! Storage abstraction for Tensors.
//!
//! # What is Storage?
//!
//! "Storage" is the container that holds the raw numerical data of a tensor.
//! While a `Tensor` struct holds metadata like shape and strides, the `Storage` holds the actual bits.
//!
//! - **Contiguous Memory**: Deep learning operations (like matrix multiplication) are fastest when
//!   data is stored contiguously in memory. This allows CPUs/GPUs to load data efficiently into caches
//!   and registers (SIMD).
//! - **Abstraction**: By defining a `Storage` trait, `xla-rs` can support multiple backends:
//!     - `Vec<T>`: Dynamic heap allocation (standard CPU tensor).
//!     - `[T; N]`: Static stack allocation (compile-time tensor).
//!     - (Future) `CudaBuffer`: GPU memory.
//!     - (Future) `Mmap`: Memory-mapped files for loading huge models.
//!
//! # The `Storage` Trait
//!
//! This trait defines the minimal interface required for a container to back a Tensor.
//! Crucially, it must provide access to the data as a **slice** (`&[T]`), which is the standard
//! Rust way to view a contiguous block of memory.

use crate::tensor::TensorElem;
use std::fmt::Debug;

/// A trait for the underlying data storage.
///
/// # Design Philosophy
///
/// We separate `Storage` from `Tensor` to allow the same high-level API to work on different
/// hardware or memory layouts.
///
/// - `T`: The element type (e.g., `f32`).
pub trait Storage<T>: Clone + Debug + Send + Sync {
    /// Returns the data as an immutable slice.
    fn as_slice(&self) -> &[T];

    /// Returns the data as a mutable slice.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Returns the number of elements in the storage.
    fn len(&self) -> usize;

    /// Returns `true` if the storage contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copies data from a slice into the storage.
    ///
    /// # Arguments
    ///
    /// * `src` - The source slice to copy from.
    fn copy_from_slice(&mut self, src: &[T])
    where
        T: Copy,
    {
        self.as_mut_slice().copy_from_slice(src);
    }
}

/// Implementation of Storage for `Vec<T>`.
///
/// This is the standard storage for CPU tensors.
/// - **Pros**: Dynamic size (can be resized), heap allocated (can be large).
/// - **Cons**: Slight allocation overhead.
impl<T: TensorElem> Storage<T> for Vec<T> {
    fn as_slice(&self) -> &[T] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
    fn len(&self) -> usize {
        self.len()
    }
}

/// Implementation of Storage for fixed-size arrays `[T; N]`.
///
/// This is used for `ConstDevice` tensors.
/// - **Pros**: Stack allocated (zero allocation overhead), size known at compile time.
/// - **Cons**: Size must be fixed at compile time, stack size limits (don't put 1GB here!).
impl<T: TensorElem, const N: usize> Storage<T> for [T; N] {
    fn as_slice(&self) -> &[T] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
    fn len(&self) -> usize {
        N
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_storage() {
        let mut storage = vec![1.0, 2.0, 3.0];

        // Test as_slice
        assert_eq!(storage.as_slice(), &[1.0, 2.0, 3.0]);

        // Test len
        assert_eq!(storage.len(), 3);
        assert!(!storage.is_empty());

        // Test as_mut_slice
        storage.as_mut_slice()[0] = 10.0;
        assert_eq!(storage.as_slice(), &[10.0, 2.0, 3.0]);

        // Test copy_from_slice
        storage.copy_from_slice(&[4.0, 5.0, 6.0]);
        assert_eq!(storage.as_slice(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_empty_storage() {
        let storage: Vec<f32> = vec![];
        assert!(storage.is_empty());
        assert_eq!(storage.len(), 0);
    }

    #[derive(Clone, Debug)]
    struct MockStorage {
        data: Vec<f32>,
    }

    impl Storage<f32> for MockStorage {
        fn as_slice(&self) -> &[f32] {
            &self.data
        }
        fn as_mut_slice(&mut self) -> &mut [f32] {
            &mut self.data
        }
        fn len(&self) -> usize {
            self.data.len()
        }
    }

    #[test]
    fn test_storage_defaults() {
        let mut storage = MockStorage {
            data: vec![1.0, 2.0],
        };
        // Test default is_empty
        assert!(!storage.is_empty());

        let empty = MockStorage { data: vec![] };
        assert!(empty.is_empty());

        // Test default copy_from_slice
        storage.copy_from_slice(&[3.0, 4.0]);
        assert_eq!(storage.as_slice(), &[3.0, 4.0]);
    }

    #[test]
    fn test_array_storage() {
        let mut storage = [1.0, 2.0, 3.0];

        // Test as_slice
        assert_eq!(storage.as_slice(), &[1.0, 2.0, 3.0]);

        // Test len
        assert_eq!(storage.len(), 3);
        assert!(!storage.is_empty());

        // Test as_mut_slice
        storage.as_mut_slice()[0] = 10.0;
        assert_eq!(storage.as_slice(), &[10.0, 2.0, 3.0]);

        // Test copy_from_slice
        storage.copy_from_slice(&[4.0, 5.0, 6.0]);
        assert_eq!(storage.as_slice(), &[4.0, 5.0, 6.0]);
    }
}
