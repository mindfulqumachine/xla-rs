use super::Optimizer;
use crate::tensor::{Cpu, Result, Tensor, TensorElem};
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
}

impl<T: TensorElem> Optimizer<T> for AdamW<T> {
    fn update<const RANK: usize>(
        &mut self,
        key: usize,
        param: &mut Tensor<T, RANK, Cpu>,
        grad: &Tensor<T, RANK, Cpu>,
    ) -> Result<()> {
        if param.shape() != grad.shape() {
            return Err(crate::tensor::TensorError::ShapeMismatch {
                expected: param.shape().to_vec(),
                got: grad.shape().to_vec(),
            });
        }

        let size = param.size();

        // Initialize state if missing
        let entry = self.state.entry(key).or_insert_with(|| {
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
        // bias_correction1 = 1 - beta1^t
        // bias_correction2 = 1 - beta2^t
        // We compute them as T.
        // Note: T::pow is not standard in Num, but we can use repeated mul or convert to f64.
        // TensorElem requires FromPrimitive/ToPrimitive.
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
        adam.update(0, &mut param, &grad).unwrap();

        // Manual check:
        // m = 0.1 * 0.1 = 0.01
        // v = 0.001 * 0.1^2 = 0.00001
        // m_hat = 0.01 / (1 - 0.9) = 0.1
        // v_hat = 0.00001 / (1 - 0.999) = 0.01
        // p = 1.0 - 0.1 * (0.1 / (sqrt(0.01) + 1e-8)) = 1.0 - 0.1 * (0.1 / 0.1) = 0.9

        let p = param.data()[0];
        assert!((p - 0.9).abs() < 1e-5, "Step 1 failed: p={}", p);

        // Step 2 (State persistence)
        adam.update(0, &mut param, &grad).unwrap();
        // Should use updated m and v
    }
}
