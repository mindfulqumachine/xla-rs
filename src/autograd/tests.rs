#[cfg(test)]
mod tests {
    use crate::tensor::{Tensor, Cpu};
    use crate::autograd::Variable;

    #[test]
    fn test_autograd_add() {
        // a = [2, 3]
        // b = [4, 5]
        // c = a + b = [6, 8]
        // d(c)/da = 1, d(c)/db = 1

        let a_data = Tensor::<f32, 1>::new(vec![2.0, 3.0], [2]).unwrap();
        let b_data = Tensor::<f32, 1>::new(vec![4.0, 5.0], [2]).unwrap();

        let a = Variable::new(a_data, true);
        let b = Variable::new(b_data, true);

        let c = &a + &b;

        c.backward();

        let ga = a.grad().unwrap();
        let gb = b.grad().unwrap();

        assert_eq!(ga.data(), &[1.0, 1.0]);
        assert_eq!(gb.data(), &[1.0, 1.0]);
    }

    #[test]
    fn test_autograd_matmul() {
        // A: [1, 2] = [[1, 2]]
        // B: [2, 1] = [[3], [4]]
        // C = A @ B = [[1*3 + 2*4]] = [[11]]

        // dC/dA = B.T = [[3, 4]]
        // dC/dB = A.T = [[1], [2]]

        let a_data = Tensor::<f32, 2>::new(vec![1.0, 2.0], [1, 2]).unwrap();
        let b_data = Tensor::<f32, 2>::new(vec![3.0, 4.0], [2, 1]).unwrap();

        let a = Variable::new(a_data, true);
        let b = Variable::new(b_data, true);

        let c = a.matmul(&b);

        c.backward();

        let ga = a.grad().unwrap();
        let gb = b.grad().unwrap();

        assert_eq!(ga.data(), &[3.0, 4.0]);
        assert_eq!(gb.data(), &[1.0, 2.0]);
    }
}
