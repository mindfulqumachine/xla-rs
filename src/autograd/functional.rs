use crate::autograd::Variable;
use crate::tensor::{Cpu, Tensor, TensorElem};

/// Computes the gradient of a function `f` with respect to its input.
///
/// Returns a function that takes a `Tensor` input and returns the gradient `Tensor`.
///
/// # Example
/// ```ignore
/// let grad_square = grad(|x| x.clone() * x.clone());
/// let g = grad_square(Tensor::new(vec![3.0], []).unwrap());
/// // g = 6.0
/// ```
pub fn grad<F, T, const RANK: usize>(f: F) -> impl Fn(Tensor<T, RANK, Cpu>) -> Tensor<T, RANK, Cpu>
where
    F: Fn(Variable<T, RANK>) -> Variable<T, RANK>,
    T: TensorElem + 'static,
{
    move |x| {
        let x_var = Variable::new(x);
        let y_var = f(x_var.clone());
        y_var.backward();

        // Extract gradient. If None (no dependency), return zeros.
        let grad = x_var.grad.borrow();
        if let Some(g) = grad.as_ref() {
            g.clone()
        } else {
            Tensor::zeros(*x_var.data.shape())
        }
    }
}

type CpuTensor<T, const RANK: usize> = Tensor<T, RANK, Cpu>;

/// Computes the value and gradient of a function `f` with respect to its input.
///
/// Returns a function that takes a `Tensor` input and returns a tuple `(Value, Gradient)`.
pub fn value_and_grad<F, T, const RANK: usize>(
    f: F,
) -> impl Fn(CpuTensor<T, RANK>) -> (CpuTensor<T, RANK>, CpuTensor<T, RANK>)
where
    F: Fn(Variable<T, RANK>) -> Variable<T, RANK>,
    T: TensorElem + 'static,
{
    move |x| {
        let x_var = Variable::new(x);
        let y_var = f(x_var.clone());
        y_var.backward();

        let grad = x_var.grad.borrow();
        let g = if let Some(g) = grad.as_ref() {
            g.clone()
        } else {
            Tensor::zeros(*x_var.data.shape())
        };

        (y_var.data, g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_square() {
        // f(x) = x^2
        // f'(x) = 2x
        let square = |x: Variable<f32, 0>| x.clone() * x.clone();
        let grad_square = grad(square);

        let x = Tensor::new(vec![3.0], []).unwrap();
        let g = grad_square(x);

        assert_eq!(g.data()[0], 6.0);
    }

    #[test]
    fn test_value_and_grad_cubic() {
        // f(x) = x^3 = x * x^2
        // f'(x) = 3x^2
        let cubic = |x: Variable<f32, 0>| x.clone() * x.clone() * x.clone();
        let vag_cubic = value_and_grad(cubic);

        let x = Tensor::new(vec![2.0], []).unwrap();
        let (val, g) = vag_cubic(x);

        assert_eq!(val.data()[0], 8.0); // 2^3
        assert_eq!(g.data()[0], 12.0); // 3 * 2^2
    }

    #[test]
    fn test_grad_constant() {
        // f(x) = 5.0
        // f'(x) = 0.0
        let constant = |_x: Variable<f32, 0>| Variable::new(Tensor::new(vec![5.0], []).unwrap());
        let grad_constant = grad(constant);

        let x = Tensor::new(vec![2.0], []).unwrap();
        let g = grad_constant(x);

        assert_eq!(g.data()[0], 0.0);
    }

    #[test]
    fn test_value_and_grad_constant() {
        // f(x) = 5.0
        let constant = |_x: Variable<f32, 0>| Variable::new(Tensor::new(vec![5.0], []).unwrap());
        let vag_constant = value_and_grad(constant);

        let x = Tensor::new(vec![2.0], []).unwrap();
        let (val, g) = vag_constant(x);

        assert_eq!(val.data()[0], 5.0);
        assert_eq!(g.data()[0], 0.0);
    }
}
