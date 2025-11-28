//! Storage abstraction for Tensors.
//!
//! This module defines the `Storage` trait, which abstracts over the underlying data container.

use crate::tensor::TensorElem;
use std::fmt::Debug;

/// A trait for the underlying data storage.
///
/// Abstracts over the container used to hold tensor data.
/// For `Cpu` device, this is typically `Vec<T>`.
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
