//! Data loading and processing utilities.
//!
//! This module provides primitives for loading, sampling, and batching data for training.
//!
//! # Components
//!
//! - **Dataset**: A trait for accessing individual data items.
//! - **Sampler**: A trait for determining the order of data access.
//! - **DataLoader**: An iterator that batches and collates data from a Dataset.

pub mod loader;
pub mod sampler;

pub use loader::DataLoader;
pub use sampler::{RandomSampler, Sampler, SequentialSampler};

/// A trait for accessing data items.
///
/// A `Dataset` represents a collection of data items (e.g., images, text samples)
/// that can be accessed by index.
///
/// # Type Parameters
///
/// * `T`: The type of the data item returned by `get`.
pub trait Dataset<T>: Send + Sync {
    /// Returns the total number of items in the dataset.
    fn len(&self) -> usize;

    /// Returns `true` if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the item at the given index.
    ///
    /// # Panics
    ///
    /// May panic if `index` is out of bounds.
    fn get(&self, index: usize) -> T;
}

// Implement Dataset for Vec<T>
impl<T: Clone + Send + Sync> Dataset<T> for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }

    fn get(&self, index: usize) -> T {
        self[index].clone()
    }
}
