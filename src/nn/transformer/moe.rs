//! # Mixture of Experts (MoE)
//!
//! This module implements a sparse Mixture of Experts (MoE) layer, a technique to scale model capacity
//! without proportionally increasing computational cost.
//!
//! ## What is Mixture of Experts?
//!
//! In a standard dense model, every input token is processed by every parameter in the network.
//! In an MoE model, the "FeedForward" block is replaced by a set of "Experts" (usually smaller FeedForward networks)
//! and a "Router" (or Gate). For each token, the Router selects a small subset (Top-K) of experts to process it.
//!
//! $$ \text{Output} = \sum_{i \in \text{TopK}} w_i \cdot \text{Expert}_i(x) $$
//!
//! This allows the model to have a massive number of parameters (high capacity) while only using a fraction
//! of them per token (low inference latency).
//!
//! ## Implementation Details
//!
//! This implementation provides:
//! - **`TopKRouter`**: A learnable gating mechanism that projects inputs to expert logits and selects the top-k indices.
//! - **`MoELayer`**: The container that holds the router and the list of experts.
//! - **`Expert` Trait**: An abstraction for what constitutes an expert (typically an MLP).
//!
//! ### Routing Mechanism
//!
//! We use a standard Top-K routing mechanism:
//! 1.  Compute logits: $H(x) = x \cdot W_{gate}$
//! 2.  Select Top-K: Identify the $k$ experts with the highest logits.
//! 3.  Normalize: Apply Softmax to the selected logits to get routing weights.
//! 4.  Dispatch: Send tokens to their respective experts.
//! 5.  Combine: Weighted sum of expert outputs.
//!
//! ## Trade-offs and Design Decisions
//!
//! ### 1. Explicit Loops vs. Scatter/Gather
//! **Decision**: We use explicit iteration and grouping (bucketing) of tokens per expert rather than
//! optimized scatter/gather tensor operations.
//!
//! **Why?**
//! - **Simplicity**: `xla-rs` is designed for clarity and education. Implementing efficient sparse scatter/gather
//!   kernels is complex and hardware-specific.
//! - **CPU Focus**: On CPU, the overhead of grouping tokens is often negligible compared to the matrix multiplications
//!   inside the experts. Explicit grouping allows us to use standard dense matrix multiplication for each expert,
//!   which is well-optimized.
//!
//! ### 2. Dynamic Control Flow
//! **Decision**: The routing logic dynamically constructs batches for each expert at runtime.
//!
//! **Why?**
//! - This avoids padding and wasted computation associated with fixed-size expert buffers (common in TPU/GPU implementations).
//! - It handles load imbalance naturally (though extreme imbalance can still hurt performance due to stragglers).
//!
//! ### 3. Generic Experts
//! **Decision**: Experts are generic modules implementing the `Expert` trait.
//!
//! **Why?**
//! - Allows experimenting with different expert architectures (e.g., different activation functions,
//!   or even nested MoEs) without changing the routing logic.

use crate::nn::{Linear, Module};
use crate::tensor::{Cpu, Result, Tensor, TensorElem};
use num_traits::Float;
use rayon::prelude::*;

/// Top-K Router for Mixture of Experts.
///
/// Routes inputs to the top-k experts based on gate logits.
#[derive(Debug)]
pub struct TopKRouter<T: TensorElem> {
    pub gate: Linear<T>,
    pub num_experts: usize,
    pub k: usize,
}

impl<T: TensorElem + Float> TopKRouter<T> {
    /// Creates a new TopKRouter.
    ///
    /// # Arguments
    ///
    /// * `gate` - The linear layer used to compute routing logits.
    /// * `num_experts` - Total number of experts.
    /// * `k` - Number of experts to route to per token.
    pub fn new(gate: Linear<T>, num_experts: usize, k: usize) -> Self {
        Self {
            gate,
            num_experts,
            k,
        }
    }

    /// Performs routing.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * `weights` - The routing weights for the top-k experts.
    /// * `indices` - The indices of the top-k experts.
    pub fn forward(
        &self,
        x: &Tensor<T, 3, Cpu>,
    ) -> Result<(Tensor<T, 3, Cpu>, Tensor<usize, 3, Cpu>)> {
        let logits = self.gate.forward(x)?;

        let [b, s, n_e] = *logits.shape();

        let mut weights_out = Tensor::zeros([b, s, self.k]);
        let mut indices_out = Tensor::zeros([b, s, self.k]);

        weights_out
            .data_mut()
            .par_chunks_mut(self.k)
            .zip(indices_out.data_mut().par_chunks_mut(self.k))
            .zip(logits.data().par_chunks(n_e))
            .for_each(|((w_row, i_row), l_row)| {
                let mut pairs: Vec<(T, usize)> = l_row
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, v)| (v, i))
                    .collect();

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

/// Mixture of Experts Layer.
///
/// Consists of a router and a set of experts.
#[derive(Debug)]
pub struct MoELayer<T: TensorElem, E: Module<T>> {
    pub router: TopKRouter<T>,
    pub experts: Vec<E>,
}

impl<T: TensorElem + Float, E: Module<T>> MoELayer<T, E> {
    /// Creates a new MoELayer.
    pub fn new(router: TopKRouter<T>, experts: Vec<E>) -> Self {
        Self { router, experts }
    }

    // Note: This forward signature assumes E (Expert) takes Tensor<T, 3> and returns Tensor<T, 3>.
    // Since Module trait is generic and doesn't enforce forward signature (Rust traits can't easily enforce generic methods with varying ranks),
    // we might need to assume E has a forward method or define a more specific Expert trait.
    // For now, we'll implement it assuming E has a forward method compatible with MLP.
    // But wait, Module trait is empty marker.
    // We should probably add `forward` to Module or create `Expert` trait.
    // For this refactor, I'll keep it simple and assume E is MLP-like but we can't call forward on generic E without a trait method.
    // So I will define a trait `Expert` here or use `Module` if I update it.
    // Let's update `Module` trait in a separate step if needed, or just define `Expert` trait here.

    // Actually, to avoid changing `Module` trait too much right now, let's define a local trait or just rely on the fact that we moved it.
    // But the plan said "Make MoELayer generic over Expert".
    // Let's define `Expert` trait in this file for now.
}

/// Trait for Experts in MoE.
///
/// Experts must implement `Module` and provide a `forward` method accepting Rank 3 tensors.
pub trait Expert<T: TensorElem>: Module<T> {
    fn forward(&self, x: &Tensor<T, 3, Cpu>) -> Result<Tensor<T, 3, Cpu>>;
}

impl<T: TensorElem + Float, E: Expert<T>> MoELayer<T, E> {
    /// Performs the forward pass of the MoE layer.
    ///
    /// Routes inputs to experts and aggregates the results.
    pub fn forward(&self, x: &Tensor<T, 3, Cpu>) -> Result<Tensor<T, 3, Cpu>> {
        let [b, s, h] = *x.shape();
        let (weights, indices) = self.router.forward(x)?;

        let mut final_output = Tensor::zeros([b, s, h]);

        let mut assignments: Vec<Vec<usize>> = vec![vec![]; self.experts.len()];
        let mut assignment_weights: Vec<Vec<T>> = vec![vec![]; self.experts.len()];

        let w_data = weights.data();
        let i_data = indices.data();

        for idx in 0..(b * s) {
            for k_i in 0..self.router.k {
                let expert_idx = i_data[idx * self.router.k + k_i];
                let weight = w_data[idx * self.router.k + k_i];
                assignments[expert_idx].push(idx);
                assignment_weights[expert_idx].push(weight);
            }
        }

        type ExpertResult<T> = Option<(Vec<usize>, Tensor<T, 2, Cpu>, Vec<T>)>;
        let results: Vec<ExpertResult<T>> = self
            .experts
            .par_iter()
            .enumerate()
            .map(|(e_idx, expert)| {
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

                Some((
                    indices.clone(),
                    output_2d,
                    assignment_weights[e_idx].clone(),
                ))
            })
            .collect();

        let out_data = final_output.data_mut();

        for (indices, output_tensor, weights) in results.into_iter().flatten() {
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

        Ok(final_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    // Mock Expert
    #[derive(Debug)]
    struct MockExpert {
        id: usize,
    }

    impl Module<f32> for MockExpert {}

    impl Expert<f32> for MockExpert {
        fn forward(&self, x: &Tensor<f32, 3, Cpu>) -> Result<Tensor<f32, 3, Cpu>> {
            // Expert returns input * id
            let scale = self.id as f32;
            let out = x.map(|v| v * scale);
            Ok(out)
        }
    }

    #[test]
    fn test_moe_forward() {
        // 2 Experts, Top-1 Routing
        // Input: [1, 2, 2] (Batch=1, Seq=2, Dim=2)
        // Router Gate: Identity-like to force routing

        // Input:
        // [[1.0, 0.0],  -> Should route to Expert 0 if gate favors index 0
        //  [0.0, 1.0]]  -> Should route to Expert 1 if gate favors index 1

        let input_data = vec![1.0, 0.0, 0.0, 1.0];
        let input = Tensor::<f32, 3, Cpu>::new(input_data, [1, 2, 2]).unwrap();

        // Gate weights: Identity [2, 2]
        // [1, 0]
        // [0, 1]
        let gate_w_data = vec![1.0, 0.0, 0.0, 1.0];
        let gate_w = Tensor::<f32, 2, Cpu>::new(gate_w_data, [2, 2]).unwrap();
        let gate = Linear::new(gate_w, None);

        let router = TopKRouter::new(gate, 2, 1);

        let experts = vec![
            MockExpert { id: 10 }, // Expert 0 scales by 10
            MockExpert { id: 20 }, // Expert 1 scales by 20
        ];

        let moe = MoELayer::new(router, experts);

        let output = moe.forward(&input).unwrap();

        // Token 0: [1, 0] -> Gate -> [1, 0] -> Top1 is Index 0 (Score 1.0).
        // Softmax([1, 0]) -> [0.73, 0.27]. Top1 weight approx 0.73?
        // Wait, TopKRouter implementation:
        // pairs sorted. top_k = pairs[0..k].
        // max_val = top_k[0].0
        // sum_exp...
        // If k=1, max_val = val. exp(val - max_val) = exp(0) = 1.
        // sum_exp = 1.
        // weight = 1/1 = 1.
        // So weight is 1.0 for top-1.

        // Token 0 routes to Expert 0 (id 10). Input [1, 0]. Output [10, 0]. Weight 1.
        // Final Token 0: [10, 0].

        // Token 1: [0, 1] -> Gate -> [0, 1] -> Top1 is Index 1 (Score 1.0).
        // Weight 1.0.
        // Token 1 routes to Expert 1 (id 20). Input [0, 1]. Output [0, 20]. Weight 1.
        // Final Token 1: [0, 20].

        let out_data = output.data();
        assert!((out_data[0] - 10.0).abs() < 1e-4);
        assert!((out_data[1] - 0.0).abs() < 1e-4);
        assert!((out_data[2] - 0.0).abs() < 1e-4);
        assert!((out_data[3] - 20.0).abs() < 1e-4);
    }
}
