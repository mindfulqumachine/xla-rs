use xla_rs::autograd::Variable;
use xla_rs::tensor::Tensor;

#[test]
fn test_autograd_simple() {
    // a = 2, b = 3
    let a_data = Tensor::<f32, 1>::new(vec![2.0], [1]).unwrap();
    let b_data = Tensor::<f32, 1>::new(vec![3.0], [1]).unwrap();

    let a = Variable::new(a_data);
    let b = Variable::new(b_data);

    // c = a * b = 6
    let c = a.clone() * b.clone();

    assert_eq!(c.data.data(), &[6.0]);

    // Backward
    c.backward();

    // da = b = 3
    let a_grad = a.grad.borrow();
    assert_eq!(a_grad.as_ref().unwrap().data(), &[3.0]);

    // db = a = 2
    let b_grad = b.grad.borrow();
    assert_eq!(b_grad.as_ref().unwrap().data(), &[2.0]);
}
