use crate::tensor::{Tensor, TensorElem, Cpu, Result};
use crate::nn::{Linear, Activation, Module};
use crate::gemma::MLP;
use rayon::prelude::*;
use num_traits::{ToPrimitive, Float};

#[derive(Debug)]
pub struct TopKRouter<T: TensorElem> {
    pub gate: Linear<T>,
    pub num_experts: usize,
    pub k: usize,
}

impl<T: TensorElem + Float> TopKRouter<T> {
    pub fn new(gate: Linear<T>, num_experts: usize, k: usize) -> Self {
        Self { gate, num_experts, k }
    }

    pub fn forward(&self, x: &Tensor<T, 3, Cpu>) -> Result<(Tensor<T, 3, Cpu>, Tensor<usize, 3, Cpu>)> {
        let logits = self.gate.forward(x)?;

        let [b, s, n_e] = *logits.shape();

        let mut weights_out = Tensor::zeros([b, s, self.k]);
        let mut indices_out = Tensor::zeros([b, s, self.k]);

        weights_out.data_mut().par_chunks_mut(self.k)
            .zip(indices_out.data_mut().par_chunks_mut(self.k))
            .zip(logits.data().par_chunks(n_e))
            .for_each(|((w_row, i_row), l_row)| {
                let mut pairs: Vec<(T, usize)> = l_row.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();

                pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                let top_k = &pairs[0..self.k];

                let max_val = top_k[0].0;
                let mut sum_exp = T::zero();
                let mut exps = Vec::with_capacity(self.k);

                for (val, _) in top_k {
                    let exp_v = (*val - max_val).to_f32().unwrap().exp();
                    let exp_v_t = T::from_f32(exp_v).unwrap();
                    sum_exp += exp_v_t;
                    exps.push(exp_v_t);
                }

                let inv_sum = T::one() / sum_exp;

                for idx in 0..self.k {
                    w_row[idx] = exps[idx] * inv_sum;
                    i_row[idx] = top_k[idx].1;
                }
            });

        Ok((weights_out, indices_out))
    }
}

#[derive(Debug)]
pub struct MoELayer<T: TensorElem> {
    pub router: TopKRouter<T>,
    pub experts: Vec<MLP<T>>,
}

impl<T: TensorElem + Float> MoELayer<T> {
    pub fn new(router: TopKRouter<T>, experts: Vec<MLP<T>>) -> Self {
        Self { router, experts }
    }

    pub fn forward(&self, x: &Tensor<T, 3, Cpu>) -> Result<Tensor<T, 3, Cpu>> {
        let [b, s, h] = *x.shape();
        let (weights, indices) = self.router.forward(x)?;

        let mut final_output = Tensor::zeros([b, s, h]);

        let mut assignments: Vec<Vec<usize>> = vec![vec![]; self.experts.len()];
        let mut assignment_weights: Vec<Vec<T>> = vec![vec![]; self.experts.len()];

        let w_data = weights.data();
        let i_data = indices.data();

        for idx in 0..(b*s) {
            for k_i in 0..self.router.k {
                let expert_idx = i_data[idx * self.router.k + k_i];
                let weight = w_data[idx * self.router.k + k_i];
                assignments[expert_idx].push(idx);
                assignment_weights[expert_idx].push(weight);
            }
        }

        let results: Vec<Option<(Vec<usize>, Tensor<T, 2, Cpu>, Vec<T>)>> = self.experts.par_iter().enumerate().map(|(e_idx, expert)| {
            let indices: &Vec<usize> = &assignments[e_idx];
            if indices.is_empty() {
                return None;
            }

            let num_samples = indices.len();
            let mut input_data = Vec::with_capacity(num_samples * h);

            for &token_idx in indices {
                let start = token_idx * h;
                input_data.extend_from_slice(&x.data()[start..start + h]);
            }

            let input_tensor = Tensor::new(input_data, [num_samples, h]).unwrap();

            let input_3d = input_tensor.reshape([num_samples, 1, h]).unwrap();

            let output_3d = expert.forward(&input_3d).unwrap();
            let output_2d = output_3d.reshape([num_samples, h]).unwrap();

            Some((indices.clone(), output_2d, assignment_weights[e_idx].clone()))
        }).collect();

        let out_data = final_output.data_mut();

        for res in results {
            if let Some((indices, output_tensor, weights)) = res {
                let out_vals = output_tensor.data();
                for (i, &token_idx) in indices.iter().enumerate() {
                    let weight = weights[i];
                    let out_offset = token_idx * h;
                    let val_offset = i * h;

                    for j in 0..h {
                        out_data[out_offset + j] += out_vals[val_offset + j] * weight;
                    }
                }
            }
        }

        Ok(final_output)
    }
}
