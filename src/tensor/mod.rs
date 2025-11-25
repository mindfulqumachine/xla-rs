use num_traits::{Float, FromPrimitive, Num, NumAssign, ToPrimitive};
use std::fmt::Debug;
use thiserror::Error;

pub mod ops;
#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        // Positive case
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::<f32, 2>::new(data.clone(), [2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), &data[..]);

        // Negative case: Size mismatch
        let err = Tensor::<f32, 2>::new(vec![1.0, 2.0, 3.0], [2, 2]);
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_zeros_ones() {
        let zeros = Tensor::<f32, 2>::zeros([2, 3]);
        assert_eq!(zeros.data(), &[0.0; 6]);

        let ones = Tensor::<f32, 2>::ones([2, 3]);
        assert_eq!(ones.data(), &[1.0; 6]);
    }

    #[test]
    fn test_reshape() {
        let tensor = Tensor::<f32, 2>::zeros([2, 3]); // 6 elements

        // Valid reshape
        let reshaped = tensor.reshape([3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);

        // Invalid reshape
        let err = reshaped.clone().reshape([4, 2]); // 8 elements
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_arithmetic() {
        let a = Tensor::<f32, 1>::new(vec![1.0, 2.0], [2]).unwrap();
        let b = Tensor::<f32, 1>::new(vec![3.0, 4.0], [2]).unwrap();

        // Add
        let c = (&a + &b).unwrap();
        assert_eq!(c.data(), &[4.0, 6.0]);

        // Mul
        let d = (&a * &b).unwrap();
        assert_eq!(d.data(), &[3.0, 8.0]);

        // Mismatch
        let e = Tensor::<f32, 1>::new(vec![1.0], [1]).unwrap();
        let f = Tensor::<f32, 1>::new(vec![1.0, 2.0, 3.0], [3]).unwrap();
        let err = &a + &f;
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_matmul_2d() {
        // A: [2, 3], B: [3, 2] -> C: [2, 2]
        let a_data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ];
        let a = Tensor::<f32, 2>::new(a_data, [2, 3]).unwrap();

        let b_data = vec![
            7.0, 8.0,
            9.0, 1.0,
            2.0, 3.0
        ];
        let b = Tensor::<f32, 2>::new(b_data, [3, 2]).unwrap();

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);

        // Row 0: 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
        // Row 0, Col 1: 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
        // Row 1: 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
        // Row 1, Col 1: 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
        assert_eq!(c.data(), &[31.0, 19.0, 85.0, 55.0]);
    }

    #[test]
    fn test_matmul_broadcast_error() {
        let a = Tensor::<f32, 2>::zeros([2, 3]);
        let b = Tensor::<f32, 2>::zeros([4, 2]); // K mismatch (3 vs 4)

        let err = a.matmul(&b);
        assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_transpose() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::<f32, 2>::new(data, [2, 3]).unwrap();
        // [ 1 2 3 ]
        // [ 4 5 6 ]

        let t_t = t.transpose().unwrap();
        assert_eq!(t_t.shape(), &[3, 2]);
        // [ 1 4 ]
        // [ 2 5 ]
        // [ 3 6 ]
        assert_eq!(t_t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_axes() {
        // Rank 4 tensor [B, S, H, D] -> [B, H, S, D]
        // Shape: [1, 2, 2, 2] -> [1, 2, 2, 2] for simplicity but distinct values
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();

        let t = Tensor::<f32, 4>::new(data, [1, 2, 2, 2]).unwrap();

        let permuted = t.transpose_axes(1, 2).unwrap();
        assert_eq!(permuted.shape(), &[1, 2, 2, 2]); // H, S swapped but sizes same

        assert_eq!(permuted.data(), &[0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
    }
}

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
    fn to_vec(&self) -> Vec<T>;

    fn copy_from_slice(&mut self, src: &[T]) where T: Copy {
        self.as_mut_slice().copy_from_slice(src);
    }
}

/// Trait bound for elements that can be stored in a Tensor.
pub trait TensorElem:
    Num + NumAssign + Copy + Clone + Debug + Send + Sync + FromPrimitive + ToPrimitive + PartialOrd + 'static
{
    /// Optional optimized matrix multiplication.
    fn gemm(m: usize, k: usize, n: usize, a: &[Self], b: &[Self]) -> Option<Vec<Self>> {
        None
    }
}

// Macro to implement TensorElem for types
macro_rules! impl_tensor_elem {
    ($($t:ty),*) => {
        $(
            impl TensorElem for $t {}
        )*
    };
}

impl_tensor_elem!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

impl TensorElem for f32 {
    fn gemm(m: usize, k: usize, n: usize, a: &[Self], b: &[Self]) -> Option<Vec<Self>> {
        let mut c = vec![0.0; m * n];
        unsafe {
            matrixmultiply::sgemm(
                m, k, n,
                1.0,
                a.as_ptr(), k as isize, 1,
                b.as_ptr(), n as isize, 1,
                0.0,
                c.as_mut_ptr(), n as isize, 1,
            );
        }
        Some(c)
    }
}

impl TensorElem for f64 {
    fn gemm(m: usize, k: usize, n: usize, a: &[Self], b: &[Self]) -> Option<Vec<Self>> {
        let mut c = vec![0.0; m * n];
        unsafe {
            matrixmultiply::dgemm(
                m, k, n,
                1.0,
                a.as_ptr(), k as isize, 1,
                b.as_ptr(), n as isize, 1,
                0.0,
                c.as_mut_ptr(), n as isize, 1,
            );
        }
        Some(c)
    }
}

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
    fn to_vec(&self) -> Vec<T> { self.clone() }
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

    pub fn to_cpu(&self) -> Tensor<T, RANK, Cpu> {
        let data_vec = self.data.to_vec();
        Tensor {
            shape: self.shape,
            strides: self.strides,
            data: data_vec,
            device: Cpu,
        }
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
