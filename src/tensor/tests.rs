use crate::tensor::{Tensor, Cpu, TensorError};

#[test]
fn test_tensor_creation() {
    // Positive case
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::<f32, 2>::new(data.clone(), [2, 2]).unwrap();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.data(), &data[..]);

    // Negative case: Size mismatch
    let err = Tensor::<f32, 2>::new(vec![1.0, 2.0, 3.0], [2, 2]);
    assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
}

#[test]
fn test_zeros_ones() {
    let zeros = Tensor::<f32, 2>::zeros([2, 3]);
    assert_eq!(zeros.data(), &[0.0; 6]);

    let ones = Tensor::<f32, 2>::ones([2, 3]);
    assert_eq!(ones.data(), &[1.0; 6]);
}

#[test]
fn test_reshape() {
    let tensor = Tensor::<f32, 2>::zeros([2, 3]); // 6 elements

    // Valid reshape
    let reshaped = tensor.reshape([3, 2]).unwrap();
    assert_eq!(reshaped.shape(), &[3, 2]);

    // Invalid reshape
    let err = reshaped.clone().reshape([4, 2]); // 8 elements
    assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
}

#[test]
fn test_arithmetic() {
    let a = Tensor::<f32, 1>::new(vec![1.0, 2.0], [2]).unwrap();
    let b = Tensor::<f32, 1>::new(vec![3.0, 4.0], [2]).unwrap();

    // Add
    let c = (&a + &b).unwrap();
    assert_eq!(c.data(), &[4.0, 6.0]);

    // Mul
    let d = (&a * &b).unwrap();
    assert_eq!(d.data(), &[3.0, 8.0]);

    // Mismatch
    let e = Tensor::<f32, 1>::new(vec![1.0], [1]).unwrap();
    // This will fail to compile due to Rank match usually?
    // No, our arithmetic is generic over rank, but runtime check for shape.
    // Wait, generic arithmetic impl requires same Rank constant.
    // Shape mismatch inside same rank:
    let f = Tensor::<f32, 1>::new(vec![1.0, 2.0, 3.0], [3]).unwrap();
    let err = (&a + &f);
    assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
}

#[test]
fn test_matmul_2d() {
    // A: [2, 3], B: [3, 2] -> C: [2, 2]
    let a_data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    ];
    let a = Tensor::<f32, 2>::new(a_data, [2, 3]).unwrap();

    let b_data = vec![
        7.0, 8.0,
        9.0, 1.0,
        2.0, 3.0
    ];
    let b = Tensor::<f32, 2>::new(b_data, [3, 2]).unwrap();

    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);

    // Row 0: 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // Row 0, Col 1: 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // Row 1: 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // Row 1, Col 1: 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    assert_eq!(c.data(), &[31.0, 19.0, 85.0, 55.0]);
}

#[test]
fn test_matmul_broadcast_error() {
    let a = Tensor::<f32, 2>::zeros([2, 3]);
    let b = Tensor::<f32, 2>::zeros([4, 2]); // K mismatch (3 vs 4)

    let err = a.matmul(&b);
    assert!(matches!(err, Err(TensorError::ShapeMismatch { .. })));
}

#[test]
fn test_transpose() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f32, 2>::new(data, [2, 3]).unwrap();
    // [ 1 2 3 ]
    // [ 4 5 6 ]

    let t_t = t.transpose().unwrap();
    assert_eq!(t_t.shape(), &[3, 2]);
    // [ 1 4 ]
    // [ 2 5 ]
    // [ 3 6 ]
    assert_eq!(t_t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_axes() {
    // Rank 4 tensor [B, S, H, D] -> [B, H, S, D]
    // Shape: [1, 2, 2, 2] -> [1, 2, 2, 2] for simplicity but distinct values
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    // 0, 1, 2, 3, 4, 5, 6, 7
    // S=0:
    //   H=0: [0, 1]
    //   H=1: [2, 3]
    // S=1:
    //   H=0: [4, 5]
    //   H=1: [6, 7]

    let t = Tensor::<f32, 4>::new(data, [1, 2, 2, 2]).unwrap();

    let permuted = t.transpose_axes(1, 2).unwrap();
    assert_eq!(permuted.shape(), &[1, 2, 2, 2]); // H, S swapped but sizes same

    // Expected Output (H outer, S inner):
    // H=0:
    //   S=0: [0, 1]
    //   S=1: [4, 5]
    // H=1:
    //   S=0: [2, 3]
    //   S=1: [6, 7]
    // Flattened: 0, 1, 4, 5, 2, 3, 6, 7

    assert_eq!(permuted.data(), &[0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
}
