use super::{Device, TensorElem, Storage};
use std::marker::PhantomData;

#[derive(Clone, Debug, PartialEq)]
pub struct Cuda;

impl Device for Cuda {
    type Storage<T> = CudaStorage<T> where T: TensorElem;
    fn name(&self) -> &'static str { "CUDA" }
}

#[derive(Clone, Debug)]
pub struct CudaStorage<T> {
    // In a real implementation, this would be Arc<CudaSlice<T>> from cudarc
    _marker: PhantomData<T>,
}

impl<T: TensorElem> Storage<T> for CudaStorage<T> {
    fn as_slice(&self) -> &[T] {
        unimplemented!("Cannot access CudaStorage as CPU slice directly")
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        unimplemented!("Cannot access CudaStorage as CPU slice directly")
    }

    fn len(&self) -> usize {
        0
    }

    fn to_vec(&self) -> Vec<T> {
        unimplemented!("CUDA to_vec not implemented yet")
    }
}
