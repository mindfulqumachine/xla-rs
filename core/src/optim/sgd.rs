use super::Optimizer;
use crate::tensor::{Result, Tensor, TensorElem};
use rayon::prelude::*;

/// Stochastic Gradient Descent (SGD) optimizer.
///
/// Updates parameters using the rule:
/// `param = param - learning_rate * grad`
pub struct Sgd<T: TensorElem> {
    pub learning_rate: T,
}

impl<T: TensorElem> Sgd<T> {
    pub fn new(learning_rate: T) -> Self {
        Self { learning_rate }
    }

    fn update_one<const RANK: usize, D: crate::tensor::Device>(
        &mut self,
        param: &mut Tensor<T, RANK, D>,
        grad: &Tensor<T, RANK, D>,
    ) -> Result<()> {
        if param.shape() != grad.shape() {
            return Err(crate::tensor::TensorError::ShapeMismatch {
                expected: param.shape().to_vec(),
                got: grad.shape().to_vec(),
            });
        }

        let lr = self.learning_rate;

        param
            .data_mut()
            .par_iter_mut()
            .zip(grad.data().par_iter())
            .for_each(|(p, g)| {
                *p -= lr * *g;
            });

        Ok(())
    }
}

impl<T: TensorElem> Optimizer<T> for Sgd<T> {
    fn update<const RANK: usize, D: crate::tensor::Device>(
        &mut self,
        params: Vec<&mut Tensor<T, RANK, D>>,
        grads: Vec<&Tensor<T, RANK, D>>,
        _key: usize,
    ) -> Result<()> {
        for (param, grad) in params.into_iter().zip(grads.into_iter()) {
            self.update_one(param, grad)?;
        }
        Ok(())
    }

    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = T::from_f32(lr).unwrap();
    }

    fn state_dict(&self) -> std::collections::HashMap<String, Tensor<T, 1, crate::tensor::Cpu>> {
        std::collections::HashMap::new()
    }

    fn load_state_dict(
        &mut self,
        _state: &std::collections::HashMap<String, Tensor<T, 1, crate::tensor::Cpu>>,
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_sgd_new() {
        let sgd = Sgd::new(0.1);
        assert_eq!(sgd.learning_rate, 0.1);
    }

    #[test]
    fn test_sgd_update() {
        let mut sgd = Sgd::new(0.1);
        let mut param = Tensor::new(vec![1.0, 2.0], [2]).unwrap();
        let grad = Tensor::new(vec![0.5, -0.5], [2]).unwrap();

        sgd.update(vec![&mut param], vec![&grad], 0).unwrap();

        // param = param - lr * grad
        // [1.0, 2.0] - 0.1 * [0.5, -0.5]
        // [1.0 - 0.05, 2.0 - (-0.05)]
        // [0.95, 2.05]
        assert!((param.data()[0] - 0.95f64).abs() < 1e-6);
        assert!((param.data()[1] - 2.05f64).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_update_shape_mismatch() {
        let mut sgd = Sgd::new(0.1);
        let mut param = Tensor::new(vec![1.0, 2.0], [2]).unwrap();
        let grad = Tensor::new(vec![0.5], [1]).unwrap();

        let result = sgd.update(vec![&mut param], vec![&grad], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sgd_set_lr() {
        let mut sgd = Sgd::new(0.1);
        assert_eq!(sgd.learning_rate, 0.1);

        sgd.set_lr(0.05);
        assert!((sgd.learning_rate - 0.05f32).abs() < 1e-6);
    }
}
