//! Storage abstraction for Tensors.
//!
//! This module defines the `Storage` trait, which abstracts over the underlying data container.
//!
//! # ML Context
//!
//! Tensors need to store their elements somewhere. "Storage" is the container that holds
//! the raw data.
//! - **Contiguous Memory**: Most tensor operations (like matrix multiplication) rely on data
//!   being stored contiguously in memory for cache efficiency and SIMD usage.
//! - **Abstraction**: By abstracting storage, we can support different backends (CPU `Vec`,
//!   GPU buffers, mmap files) without changing the high-level Tensor API.

use crate::tensor::TensorElem;
use std::fmt::Debug;

/// A trait for the underlying data storage.
///
/// Abstracts over the container used to hold tensor data.
/// For `Cpu` device, this is typically `Vec<T>`.
///
/// # Requirements
///
/// Implementations must provide access to the raw data as slices. This allows
/// efficient interaction with low-level linear algebra routines.
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
}
