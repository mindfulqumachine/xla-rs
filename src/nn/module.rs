use crate::tensor::TensorElem;
use std::fmt::Debug;

/// A Module trait for Neural Network layers.
///
/// All neural network modules (layers, models) should implement this trait.
/// Currently, it serves as a marker trait requiring `Debug`, `Send`, and `Sync`.
pub trait Module<T: TensorElem>: Debug + Send + Sync {}
