use crate::tensor::{Cpu, Result, Tensor, TensorElem};

use rayon::prelude::*;

/// Constants for Linear Layer
const WEIGHT_RANK: usize = 2;
const BIAS_RANK: usize = 1;

/// Trait to enforce allowed ranks for Linear layer forward pass.
pub trait AllowedLinearRank<const N: usize> {}
impl AllowedLinearRank<2> for () {}
impl AllowedLinearRank<3> for () {}

/// Linear Layer: y = xA^T + b
///
/// Performs a linear transformation on the input data.
/// Supports both 2D (matrix multiplication) and 3D (batched matrix multiplication) inputs.
#[derive(Debug)]
pub struct Linear<T: TensorElem> {
    pub weight: Tensor<T, WEIGHT_RANK, Cpu>,
    pub bias: Option<Tensor<T, BIAS_RANK, Cpu>>,
}

impl<T: TensorElem> Linear<T> {
    pub fn new(
        weight: Tensor<T, WEIGHT_RANK, Cpu>,
        bias: Option<Tensor<T, BIAS_RANK, Cpu>>,
    ) -> Self {
        Self { weight, bias }
    }

    pub fn forward<const RANK: usize>(
        &self,
        x: &Tensor<T, RANK, Cpu>,
    ) -> Result<Tensor<T, RANK, Cpu>>
    where
        (): AllowedLinearRank<RANK>,
    {
        // Compile-time check (redundant with trait bound but satisfies request for safer check if desired,
        // but trait bound `AllowedLinearRank` actually prevents compilation for other ranks, so strictly unnecessary
        // to assert constant. However, we'll stick to the logic that handles 2 and 3).

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
            // RANK == 2 guaranteed by AllowedLinearRank<RANK> if not 3

            // We need to cast `w_t` (Rank 2) to `Tensor<T, RANK>` unsafely to call `matmul`
            // because compiler sees generic RANK.

            let w_t_cast: &Tensor<T, RANK, Cpu> = unsafe { std::mem::transmute(&w_t) };

            let out = x.matmul(w_t_cast)?;
            let out = if let Some(b) = &self.bias {
                let shape = out.shape();
                let [_rows, cols] = [shape[0], shape[1]];

                let bias_data = b.data();
                let mut out_mut = out;
                let out_slice = out_mut.data_mut();

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
            return Err(crate::tensor::TensorError::ShapeMismatch {
                expected: vec![cols],
                got: vec![b_cols],
            });
        }

        let mut out = x.clone();

        out.data_mut().par_chunks_mut(cols).for_each(|row| {
            for (r, b) in row.iter_mut().zip(bias.data().iter()) {
                *r += *b;
            }
        });

        Ok(out)
    }
}
