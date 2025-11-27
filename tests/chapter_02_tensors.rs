use xla_rs::tensor::Tensor;

#[test]
fn test_tensor_basics() {
    // 1. Create a 2x3 tensor
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f32, 2>::new(data, [2, 3]).unwrap();

    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.strides(), &[3, 1]); // Row-major
}

#[test]
fn test_broadcasting() {
    // A: [2, 2]
    let a = Tensor::<f32, 2>::ones([2, 2]);
    // Broadcasting is not yet implemented, so we manually broadcast for now
    let b_expanded = Tensor::<f32, 2>::new(vec![1.0, 1.0, 1.0, 1.0], [2, 2]).unwrap();

    let c = (&a + &b_expanded).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data(), &[2.0, 2.0, 2.0, 2.0]);
}
