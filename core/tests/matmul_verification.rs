use xla_rs::tensor::Tensor;

#[test]
fn test_matmul_2d_verification() {
    // A: [2, 3], B: [3, 2] -> C: [2, 2]
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::<f32, 2>::new(a_data, [2, 3]).unwrap();

    let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0];
    let b = Tensor::<f32, 2>::new(b_data, [3, 2]).unwrap();

    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data(), &[31.0, 19.0, 85.0, 55.0]);
}

#[test]
fn test_matmul_3d_verification() {
    // Batch size 2
    // Batch 0: Same as 2D test
    // Batch 1: All ones
    // A: [2, 2, 3]
    // B: [2, 3, 2]
    // C: [2, 2, 2]

    let mut a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Batch 0
    // Batch 1: [1, 1, 1; 1, 1, 1] (2x3)
    a_data.extend_from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    let mut b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // Batch 0
    // Batch 1: [1, 1; 1, 1; 1, 1] (3x2)
    b_data.extend_from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    let a = Tensor::<f32, 3>::new(a_data, [2, 2, 3]).unwrap();
    let b = Tensor::<f32, 3>::new(b_data, [2, 3, 2]).unwrap();

    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2, 2]);

    let c_data = c.data();
    // Batch 0 check
    assert_eq!(&c_data[0..4], &[31.0, 19.0, 85.0, 55.0]);
    // Batch 1 check: 1*1 + 1*1 + 1*1 = 3
    assert_eq!(&c_data[4..8], &[3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn test_matmul_4d_verification() {
    // Rank 4: [Batch, Heads, M, K] x [Batch, Heads, K, N] -> [Batch, Heads, M, N]
    // Shape: [1, 2, 2, 3] x [1, 2, 3, 2] -> [1, 2, 2, 2]

    // Batch 0, Head 0:
    // A: [[1, 2, 3], [4, 5, 6]] (2x3)
    // B: [[7, 8], [9, 1], [2, 3]] (3x2)
    // Expected: [[31, 19], [85, 55]]

    // Batch 0, Head 1:
    // A: All ones (2x3)
    // B: All ones (3x2)
    // Expected: [[3, 3], [3, 3]]

    let mut a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Head 0
    a_data.extend_from_slice(&[1.0; 6]); // Head 1

    let mut b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // Head 0
    b_data.extend_from_slice(&[1.0; 6]); // Head 1

    let a = Tensor::<f32, 4>::new(a_data, [1, 2, 2, 3]).unwrap();
    let b = Tensor::<f32, 4>::new(b_data, [1, 2, 3, 2]).unwrap();

    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[1, 2, 2, 2]);

    let c_data = c.data();
    // Head 0 check
    assert_eq!(&c_data[0..4], &[31.0, 19.0, 85.0, 55.0]);
    // Head 1 check
    assert_eq!(&c_data[4..8], &[3.0, 3.0, 3.0, 3.0]);
}
