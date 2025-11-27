use crate::tensor::TensorElem;
use std::fmt::Debug;

/// A trait for the underlying data storage.
///
/// Abstracts over the container used to hold tensor data.
/// For `Cpu` device, this is typically `Vec<T>`.
pub trait Storage<T>: Clone + Debug + Send + Sync {
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

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
