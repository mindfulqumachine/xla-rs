use crate::tensor::{Storage, TensorElem};
use std::fmt::Debug;

/// A trait representing the underlying storage device for a Tensor.
///
/// Devices determine where the data is stored (e.g., CPU, GPU) and how it is accessed.
/// Currently, only `Cpu` is implemented.
pub trait Device: Clone + Debug + PartialEq + Send + Sync {
    type Storage<T>: Storage<T>
    where
        T: TensorElem;
    fn name(&self) -> &'static str;
}

/// A CPU Device.
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
