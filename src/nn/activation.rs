use crate::tensor::{Cpu, Tensor, TensorElem};
use num_traits::Float;

pub fn silu<T: TensorElem + Float>(x: T) -> T {
    let val = x.to_f32().unwrap();
    let sig = 1.0 / (1.0 + (-val).exp());
    T::from_f32(val * sig).unwrap()
}

pub struct Activation;

impl Activation {
    pub fn silu<const RANK: usize, T: TensorElem + Float>(
        x: &Tensor<T, RANK, Cpu>,
    ) -> Tensor<T, RANK, Cpu> {
        x.map(silu)
    }
}
