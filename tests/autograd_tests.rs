use xla_rs::autograd::Variable;
use xla_rs::tensor::Tensor;

#[test]
fn test_scalar_autograd() {
    // f(x) = x^2 + 3x
    // f'(x) = 2x + 3
    // Let x = 2.0
    // f(2) = 4 + 6 = 10
    // f'(2) = 4 + 3 = 7

    let x_data = Tensor::<f32, 1>::new(vec![2.0], [1]).unwrap();
    let x = Variable::new(x_data);

    // x^2 = x * x
    let x_sq = x.clone() * x.clone();

    // 3x
    let three = Variable::new(Tensor::<f32, 1>::new(vec![3.0], [1]).unwrap());
    let three_x = three * x.clone();

    // y = x^2 + 3x
    let y = x_sq + three_x;

    assert_eq!(y.data.data()[0], 10.0);

    y.backward();

    // Check grad of x
    // x contributes to x_sq (2x) and three_x (3)
    // grad should be 7.0

    let x_grad = x.grad.borrow();
    assert!(x_grad.is_some());
    assert_eq!(x_grad.as_ref().unwrap().data()[0], 7.0);
}

#[test]
fn test_matmul_autograd() {
    // C = A @ B
    // A: [1, 2] = [1, 2]
    // B: [2, 1] = [3, 4]
    // C: [1, 1] = 1*3 + 2*4 = 3 + 8 = 11

    // dC/dA = B^T = [3, 4]
    // dC/dB = A^T = [1, 2]

    let a_data = Tensor::<f32, 2>::new(vec![1.0, 2.0], [1, 2]).unwrap();
    let b_data = Tensor::<f32, 2>::new(vec![3.0, 4.0], [2, 1]).unwrap();

    let a = Variable::new(a_data);
    let b = Variable::new(b_data);

    let c = a.matmul(&b).unwrap();

    assert_eq!(c.data.data()[0], 11.0);

    c.backward();

    let a_grad = a.grad.borrow();
    let b_grad = b.grad.borrow();

    assert!(a_grad.is_some());
    assert!(b_grad.is_some());

    // Check A grad: should be B^T = [3, 4]
    assert_eq!(a_grad.as_ref().unwrap().data(), &[3.0, 4.0]);

    // Check B grad: should be A^T = [1, 2]
    assert_eq!(b_grad.as_ref().unwrap().data(), &[1.0, 2.0]);
}
