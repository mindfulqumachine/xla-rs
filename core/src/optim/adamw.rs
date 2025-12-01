use super::Optimizer;
use crate::tensor::{Result, Tensor, TensorElem};
use rayon::prelude::*;
use std::collections::HashMap;

/// AdamW optimizer.
///
/// Implements Adam algorithm with Weight Decay fix as described in [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
///
/// # Formula
///
/// $$
/// \begin{aligned}
/// & m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
/// & v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
/// & \hat{m}_t = m_t / (1 - \beta_1^t) \\
/// & \hat{v}_t = v_t / (1 - \beta_2^t) \\
/// & \theta_t = \theta_{t-1} - \eta (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda \theta_{t-1})
/// \end{aligned}
/// $$
pub struct AdamW<T: TensorElem> {
    pub learning_rate: T,
    pub beta1: T,
    pub beta2: T,
    pub epsilon: T,
    pub weight_decay: T,
    /// State: Key -> (m, v, step)
    /// We store m and v as flat vectors to handle arbitrary tensor shapes.
    state: HashMap<usize, (Vec<T>, Vec<T>, u64)>,
}

impl<T: TensorElem> AdamW<T> {
    /// Creates a new AdamW optimizer.
    ///
    /// # Arguments
    /// * `learning_rate` - The learning rate ($\eta$).
    /// * `beta1` - Coefficient for computing running averages of gradient (default: 0.9).
    /// * `beta2` - Coefficient for computing running averages of square of gradient (default: 0.999).
    /// * `epsilon` - Term added to the denominator to improve numerical stability (default: 1e-8).
    /// * `weight_decay` - Weight decay coefficient ($\lambda$) (default: 0.01).
    pub fn new(learning_rate: T) -> Self {
        Self {
            learning_rate,
            beta1: T::from_f64(0.9).unwrap(),
            beta2: T::from_f64(0.999).unwrap(),
            epsilon: T::from_f64(1e-8).unwrap(),
            weight_decay: T::from_f64(0.01).unwrap(),
            state: HashMap::new(),
        }
    }

    pub fn with_betas(mut self, beta1: T, beta2: T) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    fn update_one<const RANK: usize, D: crate::tensor::Device>(
        &mut self,
        param: &mut Tensor<T, RANK, D>,
        grad: &Tensor<T, RANK, D>,
        key: usize,
        idx: usize,
    ) -> Result<()> {
        if param.shape() != grad.shape() {
            return Err(crate::tensor::TensorError::ShapeMismatch {
                expected: param.shape().to_vec(),
                got: grad.shape().to_vec(),
            });
        }

        let size = param.size();

        // Initialize state if missing
        // Combine key and idx to get unique ID
        let state_key = (key << 32) | idx;

        let entry = self.state.entry(state_key).or_insert_with(|| {
            (
                vec![T::zero(); size], // m
                vec![T::zero(); size], // v
                0,                     // step
            )
        });

        let (m, v, step) = entry;
        *step += 1;

        let lr = self.learning_rate;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.epsilon;
        let lambda = self.weight_decay;
        let one = T::one();

        // Bias correction terms
        let b1_t = b1.to_f64().unwrap().powi(*step as i32);
        let b2_t = b2.to_f64().unwrap().powi(*step as i32);
        let bias_correction1 = T::from_f64(1.0 - b1_t).unwrap();
        let bias_correction2 = T::from_f64(1.0 - b2_t).unwrap();

        // Parallel update
        param
            .data_mut()
            .par_iter_mut()
            .zip(grad.data().par_iter())
            .zip(m.par_iter_mut())
            .zip(v.par_iter_mut())
            .for_each(|(((p, g), m_elem), v_elem)| {
                // Update m: m = b1 * m + (1 - b1) * g
                *m_elem = b1 * *m_elem + (one - b1) * *g;

                // Update v: v = b2 * v + (1 - b2) * g^2
                *v_elem = b2 * *v_elem + (one - b2) * *g * *g;

                // Bias correction
                let m_hat = *m_elem / bias_correction1;
                let v_hat = *v_elem / bias_correction2;

                // Update param
                // p = p - lr * (m_hat / (sqrt(v_hat) + eps) + lambda * p)
                let v_sqrt = v_hat.to_f64().unwrap().sqrt();
                let denom = T::from_f64(v_sqrt).unwrap() + eps;

                *p = *p - lr * (m_hat / denom + lambda * *p);
            });

        Ok(())
    }
}

impl<T: TensorElem> Optimizer<T> for AdamW<T> {
    fn update<const RANK: usize, D: crate::tensor::Device>(
        &mut self,
        params: Vec<&mut Tensor<T, RANK, D>>,
        grads: Vec<&Tensor<T, RANK, D>>,
        key: usize,
    ) -> Result<()> {
        for (i, (param, grad)) in params.into_iter().zip(grads.into_iter()).enumerate() {
            self.update_one(param, grad, key, i)?;
        }
        Ok(())
    }

    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = T::from_f32(lr).unwrap();
    }

    fn state_dict(&self) -> std::collections::HashMap<String, Tensor<T, 1, crate::tensor::Cpu>> {
        let mut dict = HashMap::new();
        for (key, (m, v, step)) in &self.state {
            // Key format: "state.{key}.m", "state.{key}.v", "state.{key}.step"
            // m and v are Vec<T>. We need to convert to Tensor<T, 1, Cpu>.
            // step is u64. We store as Tensor<T, 1, Cpu> (size 1).

            let m_tensor = Tensor::new(m.clone(), [m.len()]).unwrap();
            let v_tensor = Tensor::new(v.clone(), [v.len()]).unwrap();
            let step_tensor = Tensor::new(vec![T::from_u64(*step).unwrap()], [1]).unwrap();

            dict.insert(format!("state.{}.m", key), m_tensor);
            dict.insert(format!("state.{}.v", key), v_tensor);
            dict.insert(format!("state.{}.step", key), step_tensor);
        }
        dict
    }

    fn load_state_dict(
        &mut self,
        state: &std::collections::HashMap<String, Tensor<T, 1, crate::tensor::Cpu>>,
    ) -> Result<()> {
        // We need to reconstruct self.state from the flat dict.
        // Keys are "state.{key}.m", etc.
        // We can iterate over keys and parse.

        // Group by key first
        type AdamWGroupedState<T> = (Option<Vec<T>>, Option<Vec<T>>, Option<u64>);
        let mut grouped_state: HashMap<usize, AdamWGroupedState<T>> = HashMap::new();

        for (name, tensor) in state {
            if name.starts_with("state.") {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() == 3 {
                    #[allow(clippy::collapsible_if)]
                    if let Ok(key) = parts[1].parse::<usize>() {
                        let field = parts[2];
                        let entry = grouped_state.entry(key).or_insert((None, None, None));

                        match field {
                            "m" => entry.0 = Some(tensor.data().to_vec()),
                            "v" => entry.1 = Some(tensor.data().to_vec()),
                            "step" => {
                                let val = tensor.data()[0];
                                entry.2 = Some(val.to_u64().unwrap());
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        self.state.clear();
        for (key, (m, v, step)) in grouped_state {
            if let (Some(m), Some(v), Some(step)) = (m, v, step) {
                self.state.insert(key, (m, v, step));
            } else {
                // Incomplete state for key, warn or error?
                // For now, skip.
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_adamw_new() {
        let adam = AdamW::<f32>::new(0.001);
        assert_eq!(adam.learning_rate, 0.001);
        assert_eq!(adam.beta1, 0.9);
        assert_eq!(adam.weight_decay, 0.01);
    }

    #[test]
    fn test_adamw_update() {
        let mut adam = AdamW::<f32>::new(0.1).with_weight_decay(0.0); // Disable WD for simple check
        let mut param = Tensor::new(vec![1.0], [1]).unwrap();
        let grad = Tensor::new(vec![0.1], [1]).unwrap();

        // Step 1
        adam.update(vec![&mut param], vec![&grad], 0).unwrap();

        // Manual check:
        // m = 0.1 * 0.1 = 0.01
        // v = 0.001 * 0.1^2 = 0.00001
        // m_hat = 0.01 / (1 - 0.9) = 0.1
        // v_hat = 0.00001 / (1 - 0.999) = 0.01
        // p = 1.0 - 0.1 * (0.1 / (sqrt(0.01) + 1e-8)) = 1.0 - 0.1 * (0.1 / 0.1) = 0.9

        let p = param.data()[0];
        assert!((p - 0.9f32).abs() < 1e-5, "Step 1 failed: p={}", p);

        // Step 2 (State persistence)
        adam.update(vec![&mut param], vec![&grad], 0).unwrap();
        // Should use updated m and v
    }
}
