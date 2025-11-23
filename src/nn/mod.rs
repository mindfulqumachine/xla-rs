use crate::tensor::{Tensor, TensorElem, Cpu, Result};
use std::fmt::Debug;
use num_traits::Float;

/// A Module trait for Neural Network layers.
pub trait Module<T: TensorElem>: Debug + Send + Sync {
}

/// Linear Layer: y = xA^T + b
#[derive(Debug)]
pub struct Linear<T: TensorElem> {
    pub weight: Tensor<T, 2, Cpu>,
    pub bias: Option<Tensor<T, 1, Cpu>>,
}

impl<T: TensorElem> Linear<T> {
    pub fn new(weight: Tensor<T, 2, Cpu>, bias: Option<Tensor<T, 1, Cpu>>) -> Self {
        Self { weight, bias }
    }

    pub fn forward<const RANK: usize>(&self, x: &Tensor<T, RANK, Cpu>) -> Result<Tensor<T, RANK, Cpu>> {
        let w_t = self.weight.transpose()?;

        if RANK == 3 {
             let [b, s, i] = x.shape()[0..3].try_into().unwrap();
             // Explicitly specify type for flat_x
             let flat_x: Tensor<T, 2, Cpu> = x.clone().reshape([b * s, i])?;
             let out_flat = flat_x.matmul(&w_t)?;

             let out_biased = if let Some(b_bias) = &self.bias {
                 Self::add_bias(&out_flat, b_bias)?
             } else {
                 out_flat
             };

             let out_features = self.weight.shape()[0];
             let res = out_biased.reshape([b, s, out_features])?;

             unsafe {
                 let res_ptr = &res as *const Tensor<T, 3, Cpu>;
                 let ret = std::ptr::read(res_ptr as *const Tensor<T, RANK, Cpu>);
                 std::mem::forget(res);
                 Ok(ret)
             }
        } else {
             // Here we assume RANK == 2.
             // But we cannot prove it to the compiler.
             // We need to cast `w_t` (Rank 2) to `Tensor<T, RANK>` unsafely to call `matmul`.
             // This is dangerous if RANK != 2, so we assert.

             if RANK != 2 {
                 return Err(crate::tensor::TensorError::Unsupported(format!("Linear support Rank 2 or 3, got {}", RANK)));
             }

             let w_t_cast: &Tensor<T, RANK, Cpu> = unsafe {
                 std::mem::transmute(&w_t)
             };

             let out = x.matmul(w_t_cast)?;
             let out = if let Some(b) = &self.bias {
                  let shape = out.shape();
                  let [_rows, cols] = [shape[0], shape[1]];

                  let bias_data = b.data();
                  let mut out_mut = out;
                  let out_slice = out_mut.data_mut();

                  use rayon::prelude::*;
                  out_slice.par_chunks_mut(cols).for_each(|row| {
                      for (r, bv) in row.iter_mut().zip(bias_data.iter()) {
                          *r += *bv;
                      }
                  });
                  out_mut
             } else {
                 out
             };
             Ok(out)
        }
    }

    fn add_bias(x: &Tensor<T, 2, Cpu>, bias: &Tensor<T, 1, Cpu>) -> Result<Tensor<T, 2, Cpu>> {
        let [_, cols] = *x.shape();
        let [b_cols] = *bias.shape();

        if cols != b_cols {
             return Err(crate::tensor::TensorError::ShapeMismatch { expected: vec![cols], got: vec![b_cols] });
        }

        use rayon::prelude::*;
        let mut out = x.clone();

        out.data_mut().par_chunks_mut(cols)
           .for_each(|row| {
               for (r, b) in row.iter_mut().zip(bias.data().iter()) {
                   *r += *b;
               }
           });

        Ok(out)
    }
}

/// RMSNorm
#[derive(Debug)]
pub struct RMSNorm<T: TensorElem> {
    pub weight: Tensor<T, 1, Cpu>,
    pub eps: T,
}

impl<T: TensorElem + Float> RMSNorm<T> {
    pub fn new(weight: Tensor<T, 1, Cpu>, eps: T) -> Self {
        Self { weight, eps }
    }

    pub fn forward<const RANK: usize>(&self, x: &Tensor<T, RANK, Cpu>) -> Result<Tensor<T, RANK, Cpu>> {
        let shape = x.shape();
        let last_dim = shape[RANK - 1];
        if last_dim != self.weight.shape()[0] {
             return Err(crate::tensor::TensorError::ShapeMismatch {
                 expected: vec![last_dim],
                 got: vec![self.weight.shape()[0]]
             });
        }

        let mut out = Tensor::zeros(*shape);

        use rayon::prelude::*;

        out.data_mut().par_chunks_mut(last_dim)
           .zip(x.data().par_chunks(last_dim))
           .for_each(|(out_row, in_row)| {
               let mut sum_sq = T::zero();
               for &val in in_row {
                   sum_sq += val * val;
               }
               let mean_sq = sum_sq / T::from_usize(last_dim).unwrap();
               let rsqrt = T::one() / (mean_sq + self.eps).sqrt();

               for i in 0..last_dim {
                   out_row[i] = in_row[i] * rsqrt * self.weight.data()[i];
               }
           });

        Ok(out)
    }
}

pub fn silu<T: TensorElem + Float>(x: T) -> T {
    let val = x.to_f32().unwrap();
    let sig = 1.0 / (1.0 + (-val).exp());
    T::from_f32(val * sig).unwrap()
}

pub struct Activation;

impl Activation {
    pub fn silu<const RANK: usize, T: TensorElem + Float>(x: &Tensor<T, RANK, Cpu>) -> Tensor<T, RANK, Cpu> {
        x.map(silu)
    }
}
