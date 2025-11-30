use super::Optimizer;
use crate::tensor::{Cpu, Result, Tensor, TensorElem};
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
}

impl<T: TensorElem> Optimizer<T> for Sgd<T> {
    fn step(
        &mut self,
        _params: &mut [&mut Tensor<T, 2, Cpu>],
        _grads: &[&Tensor<T, 2, Cpu>],
    ) -> Result<()> {
        // Deprecated/Unused in favor of `update`
        Ok(())
    }

    fn update<const RANK: usize>(
        &self,
        param: &mut Tensor<T, RANK, Cpu>,
        grad: &Tensor<T, RANK, Cpu>,
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
                *p = *p - lr * *g;
            });

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
        let sgd = Sgd::new(0.1);
        let mut param = Tensor::new(vec![1.0, 2.0], [2]).unwrap();
        let grad = Tensor::new(vec![0.5, -0.5], [2]).unwrap();

        sgd.update(&mut param, &grad).unwrap();

        // param = param - lr * grad
        // [1.0, 2.0] - 0.1 * [0.5, -0.5]
        // [1.0 - 0.05, 2.0 - (-0.05)]
        // [0.95, 2.05]
        assert!((param.data()[0] - 0.95f64).abs() < 1e-6);
        assert!((param.data()[1] - 2.05f64).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_update_shape_mismatch() {
        let sgd = Sgd::new(0.1);
        let mut param = Tensor::new(vec![1.0, 2.0], [2]).unwrap();
        let grad = Tensor::new(vec![0.5], [1]).unwrap();

        let result = sgd.update(&mut param, &grad);
        assert!(result.is_err());
    }
}
