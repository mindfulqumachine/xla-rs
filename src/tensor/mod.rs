use num_traits::{Float, FromPrimitive, Num, NumAssign, ToPrimitive};
use std::fmt::Debug;
use thiserror::Error;

pub mod ops;

/// Error type for Tensor operations.
#[derive(Error, Debug)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    #[error("Incompatible shapes for broadcasting: {0:?} and {1:?}")]
    BroadcastError(Vec<usize>, Vec<usize>),
    #[error("Index out of bounds: index {index:?} for shape {shape:?}")]
    IndexOutOfBounds { index: Vec<usize>, shape: Vec<usize> },
    #[error("Device mismatch")]
    DeviceMismatch,
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, TensorError>;

/// A trait representing the underlying storage device for a Tensor.
pub trait Device: Clone + Debug + PartialEq + Send + Sync {
    type Storage<T>: Storage<T> where T: TensorElem;
    fn name(&self) -> &'static str;
}

/// A trait for the underlying data storage.
pub trait Storage<T>: Clone + Debug + Send + Sync {
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }

    fn copy_from_slice(&mut self, src: &[T]) where T: Copy {
        self.as_mut_slice().copy_from_slice(src);
    }
}

/// Trait bound for elements that can be stored in a Tensor.
///
/// # Requirements
/// - `Copy + Clone`: Essential for efficient storage in contiguous memory (e.g., `Vec<T>`) and fast element access.
/// - `'static`: Simplifies the type system by avoiding complex lifetime management in generic Tensor structs, assuming data ownership.
/// - `Num + ...`: Provides necessary numeric operations for tensor math.
/// - `Send + Sync`: Required for parallel execution via `rayon`.
pub trait TensorElem:
    Num + NumAssign + Copy + Clone + Debug + Send + Sync + FromPrimitive + ToPrimitive + PartialOrd + 'static
{}

impl<T> TensorElem for T where
    T: Num + NumAssign + Copy + Clone + Debug + Send + Sync + FromPrimitive + ToPrimitive + PartialOrd + 'static
{}

/// A CPU Device.
#[derive(Clone, Debug, PartialEq)]
pub struct Cpu;

impl Device for Cpu {
    type Storage<T> = Vec<T> where T: TensorElem;
    fn name(&self) -> &'static str { "CPU" }
}

impl<T: TensorElem> Storage<T> for Vec<T> {
    fn as_slice(&self) -> &[T] { self }
    fn as_mut_slice(&mut self) -> &mut [T] { self }
    fn len(&self) -> usize { self.len() }
}

/// The core Tensor struct.
#[derive(Clone)]
pub struct Tensor<T, const RANK: usize, D: Device = Cpu>
where T: TensorElem {
    shape: [usize; RANK],
    strides: [usize; RANK],
    data: D::Storage<T>,
    device: D,
}

impl<T, const RANK: usize> Tensor<T, RANK, Cpu>
where T: TensorElem {
    pub fn new(data: Vec<T>, shape: [usize; RANK]) -> Result<Self> {
        let size: usize = shape.iter().product();
        if data.len() != size {
            return Err(TensorError::ShapeMismatch {
                expected: vec![size],
                got: vec![data.len()]
            });
        }

        let strides = compute_strides(&shape);
        Ok(Self {
            shape,
            strides,
            data,
            device: Cpu,
        })
    }

    pub fn zeros(shape: [usize; RANK]) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![T::zero(); size];
        let strides = compute_strides(&shape);
        Self {
            shape,
            strides,
            data,
            device: Cpu,
        }
    }

    pub fn ones(shape: [usize; RANK]) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![T::one(); size];
        let strides = compute_strides(&shape);
        Self {
            shape,
            strides,
            data,
            device: Cpu,
        }
    }

    pub fn reshape<const NEW_RANK: usize>(self, new_shape: [usize; NEW_RANK]) -> Result<Tensor<T, NEW_RANK, Cpu>> {
        let current_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();

        if current_size != new_size {
             return Err(TensorError::ShapeMismatch {
                expected: vec![current_size],
                got: vec![new_size]
            });
        }

        let strides = compute_strides(&new_shape);
        Ok(Tensor {
            shape: new_shape,
            strides,
            data: self.data,
            device: self.device
        })
    }
}

fn compute_strides<const RANK: usize>(shape: &[usize; RANK]) -> [usize; RANK] {
    let mut strides = [0; RANK];
    let mut stride = 1;
    for i in (0..RANK).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

impl<T, const RANK: usize, D: Device> Tensor<T, RANK, D>
where T: TensorElem {
    pub fn shape(&self) -> &[usize; RANK] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize; RANK] {
        &self.strides
    }

    pub fn data(&self) -> &[T] {
        self.data.as_slice()
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<T, const RANK: usize, D: Device> Debug for Tensor<T, RANK, D>
where T: TensorElem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
         .field("shape", &self.shape)
         .field("device", &self.device.name())
         .field("data_len", &self.data.len())
         .finish()
    }
}
