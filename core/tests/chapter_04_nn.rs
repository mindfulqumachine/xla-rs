use xla_rs::nn::Linear;
use xla_rs::tensor::Tensor;

#[test]
fn test_linear_forward() {
    // Weight: [2, 2] (all ones)
    let weight = Tensor::<f32, 2>::ones([2, 2]);
    // Bias: [2] (all ones)
    let bias = Tensor::<f32, 1>::ones([2]);

    let linear = Linear::new(weight, Some(bias));

    // Input: [1, 2] (all ones)
    let input = Tensor::<f32, 2>::ones([1, 2]);

    // Output = Input * Weight^T + Bias
    // [1, 1] * [1, 1]^T + [1, 1]
    // [1*1 + 1*1] + 1 = 3

    let output = linear.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(output.data(), &[3.0, 3.0]);
}
