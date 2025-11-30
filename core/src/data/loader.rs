//! DataLoader implementation.

use super::{Dataset, RandomSampler, Sampler, SequentialSampler};
use std::sync::Arc;

/// Trait for collating a list of items into a batch.
pub trait Collate<T> {
    /// The type of the batched output (e.g., `Tensor`, `Vec<Tensor>`).
    type Output;

    /// Collates a vector of items into a single batch.
    fn collate(batch: Vec<T>) -> Self::Output;
}

/// A default collator that just returns the `Vec<T>`.
/// Useful for simple cases or when manual collation is desired later.
pub struct DefaultCollate;

impl<T> Collate<T> for DefaultCollate {
    type Output = Vec<T>;

    fn collate(batch: Vec<T>) -> Self::Output {
        batch
    }
}

/// Data loader.
///
/// Combines a dataset and a sampler, and provides an iterable over the given dataset.
///
/// # Type Parameters
///
/// * `D`: The dataset type.
/// * `T`: The item type returned by the dataset.
/// * `C`: The collator type (defaults to `DefaultCollate`).
pub struct DataLoader<D, T, C = DefaultCollate>
where
    D: Dataset<T>,
    C: Collate<T>,
{
    dataset: Arc<D>,
    batch_size: usize,
    sampler: Box<dyn Sampler>,
    drop_last: bool,
    _marker: std::marker::PhantomData<(T, C)>,
}

impl<D, T> DataLoader<D, T, DefaultCollate>
where
    D: Dataset<T> + 'static,
{
    /// Creates a new DataLoader with default collation.
    pub fn new(dataset: D, batch_size: usize) -> Self {
        Self {
            dataset: Arc::new(dataset),
            batch_size,
            sampler: Box::new(SequentialSampler),
            drop_last: false,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<D, T, C> DataLoader<D, T, C>
where
    D: Dataset<T> + 'static,
    C: Collate<T>,
{
    /// Sets the sampler to use.
    pub fn with_sampler(mut self, sampler: impl Sampler + 'static) -> Self {
        self.sampler = Box::new(sampler);
        self
    }

    /// Sets whether to shuffle the data (uses `RandomSampler`).
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        if shuffle {
            self.sampler = Box::new(RandomSampler);
        } else {
            self.sampler = Box::new(SequentialSampler);
        }
        self
    }

    /// Sets whether to drop the last incomplete batch.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Returns an iterator over the dataset.
    pub fn iter(&self) -> DataLoaderIter<D, T, C> {
        let indices = self.sampler.sample(self.dataset.len());
        DataLoaderIter {
            dataset: self.dataset.clone(),
            indices,
            batch_size: self.batch_size,
            current_idx: 0,
            drop_last: self.drop_last,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<D, T, C> IntoIterator for DataLoader<D, T, C>
where
    D: Dataset<T> + 'static,
    C: Collate<T>,
{
    type Item = C::Output;
    type IntoIter = DataLoaderIter<D, T, C>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator for DataLoader.
pub struct DataLoaderIter<D, T, C>
where
    D: Dataset<T>,
    C: Collate<T>,
{
    dataset: Arc<D>,
    indices: Vec<usize>,
    batch_size: usize,
    current_idx: usize,
    drop_last: bool,
    _marker: std::marker::PhantomData<(T, C)>,
}

impl<D, T, C> Iterator for DataLoaderIter<D, T, C>
where
    D: Dataset<T>,
    C: Collate<T>,
{
    type Item = C::Output;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let remaining = self.indices.len() - self.current_idx;
        if self.drop_last && remaining < self.batch_size {
            return None;
        }

        let take = remaining.min(self.batch_size);
        let batch_indices = &self.indices[self.current_idx..self.current_idx + take];
        self.current_idx += take;

        let batch: Vec<T> = batch_indices
            .iter()
            .map(|&idx| self.dataset.get(idx))
            .collect();

        Some(C::collate(batch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader_sequential() {
        let data = vec![1, 2, 3, 4, 5];
        let loader = DataLoader::new(data, 2);

        let mut iter = loader.iter();
        assert_eq!(iter.next(), Some(vec![1, 2]));
        assert_eq!(iter.next(), Some(vec![3, 4]));
        assert_eq!(iter.next(), Some(vec![5]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_dataloader_drop_last() {
        let data = vec![1, 2, 3, 4, 5];
        let loader = DataLoader::new(data, 2).drop_last(true);

        let mut iter = loader.iter();
        assert_eq!(iter.next(), Some(vec![1, 2]));
        assert_eq!(iter.next(), Some(vec![3, 4]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_dataloader_shuffle() {
        let data = vec![1, 2, 3, 4, 5];
        let loader = DataLoader::new(data.clone(), 5).shuffle(true);

        let batch = loader.iter().next().unwrap();
        assert_eq!(batch.len(), 5);
        assert_ne!(batch, data); // Probability of collision is 1/120, negligible

        // Check content is same
        let mut sorted_batch = batch.clone();
        sorted_batch.sort();
        assert_eq!(sorted_batch, data);
    }
}
