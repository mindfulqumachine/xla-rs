use num_traits::{FromPrimitive, Num, NumAssign, ToPrimitive};
use std::fmt::Debug;
use thiserror::Error;

pub mod cpu_matmul;
pub mod cpu_transpose;

pub use cpu_matmul::cpu_matmul;
pub use cpu_transpose::cpu_transpose;

#[derive(Error, Debug)]
pub enum KernelError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

pub type Result<T> = std::result::Result<T, KernelError>;

/// Trait bound for elements that can be processed by kernels.
/// This mirrors `TensorElem` in the main crate to avoid circular dependencies.
pub trait KernelElem:
    Num + NumAssign + Copy + Clone + Debug + Send + Sync + FromPrimitive + ToPrimitive + PartialOrd
{
}

impl<T> KernelElem for T where
    T: Num
        + NumAssign
        + Copy
        + Clone
        + Debug
        + Send
        + Sync
        + FromPrimitive
        + ToPrimitive
        + PartialOrd
{
}
