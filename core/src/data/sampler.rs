//! Strategies for sampling indices from a dataset.

use rand::seq::SliceRandom;

/// A trait for determining the order of data access.
pub trait Sampler: Send + Sync {
    /// Returns an iterator over indices.
    ///
    /// # Arguments
    ///
    /// * `len`: The length of the dataset.
    fn sample(&self, len: usize) -> Vec<usize>;
}

/// Samples elements sequentially, always in the same order.
pub struct SequentialSampler;

impl Sampler for SequentialSampler {
    fn sample(&self, len: usize) -> Vec<usize> {
        (0..len).collect()
    }
}

/// Samples elements randomly (without replacement).
pub struct RandomSampler;

impl Sampler for RandomSampler {
    fn sample(&self, len: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..len).collect();
        let mut rng = rand::rng();
        indices.shuffle(&mut rng);
        indices
    }
}
