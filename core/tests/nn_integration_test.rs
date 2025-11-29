#![allow(unused)]
use std::cell::RefCell;
use std::rc::Rc;
use xla_rs::autograd::{GraphNode, Variable};
use xla_rs::tensor::{Cpu, Tensor, TensorElem};

// --- Sigmoid Activation ---

#[derive(Debug)]
struct SigmoidNode<T: TensorElem, const RANK: usize> {
    input: Tensor<T, RANK, Cpu>,
    output: Tensor<T, RANK, Cpu>,
    input_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    out_grad: Rc<RefCell<Option<Tensor<T, RANK, Cpu>>>>,
    parents: Vec<Rc<dyn GraphNode>>,
}

impl<T: TensorElem + num_traits::Float, const RANK: usize> GraphNode for SigmoidNode<T, RANK> {
    fn backward(&self) {
        if let Some(grad) = self.out_grad.borrow().as_ref() {
            // dL/dx = dL/dy * y * (1 - y)
            let one = T::one();
            let y = &self.output;

            // y * (1 - y)
            // We need to do this element-wise.
            // Since we don't have extensive tensor ops, we'll use map.

            let derivative = y.map(|val| val * (one - val));
            let dx = (grad * &derivative).unwrap();

            let mut input_grad = self.input_grad.borrow_mut();
            if let Some(g) = input_grad.as_mut() {
                *g = (g as &Tensor<T, RANK, Cpu> + &dx).unwrap(); // Add to existing gradient
            } else {
                *input_grad = Some(dx);
            }
        }
    }

    fn parents(&self) -> Vec<Rc<dyn GraphNode>> {
        self.parents.clone()
    }
}

fn sigmoid<const RANK: usize>(x: &Variable<f32, RANK>) -> Variable<f32, RANK> {
    let data = x.data.map(|v| 1.0 / (1.0 + (-v).exp()));

    let mut parents = Vec::new();
    if let Some(p) = &x.node {
        parents.push(p.clone());
    }

    let out_grad = Rc::new(RefCell::new(None));

    let node = Rc::new(SigmoidNode {
        input: x.data.clone(),
        output: data.clone(),
        input_grad: x.grad.clone(),
        out_grad: out_grad.clone(),
        parents,
    });

    Variable {
        data,
        grad: out_grad,
        node: Some(node),
    }
}

// --- Linear Layer ---

struct Linear {
    weight: Variable<f32, 2>,
    bias: Variable<f32, 2>,
}

impl Linear {
    fn new(in_features: usize, out_features: usize) -> Self {
        // Pseudo-random initialization
        let w_data = (0..in_features * out_features)
            .map(|i| {
                let s = (i + 1) as f32;
                ((s * 12.9898).sin() * 43758.5453).fract() - 0.5
            })
            .collect();
        let weight = Variable::new(Tensor::new(w_data, [in_features, out_features]).unwrap());

        // Bias shape [1, out_features] for easy addition with batch_size=1
        let b_data = vec![0.0; out_features];
        let bias = Variable::new(Tensor::new(b_data, [1, out_features]).unwrap());

        Self { weight, bias }
    }

    fn forward(&self, x: &Variable<f32, 2>) -> Variable<f32, 2> {
        // y = x @ W + b
        // x: [B, In], W: [In, Out] -> [B, Out]
        let xw = x.matmul(&self.weight).unwrap();
        xw + self.bias.clone()
    }
}

#[test]
fn test_xor_training() {
    // XOR Problem
    // Inputs: (0,0), (0,1), (1,0), (1,1)
    // Targets: 0, 1, 1, 0

    // We'll use inputs with an extra 1.0 for bias:
    // (0,0,1), (0,1,1), (1,0,1), (1,1,1)

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    // Network: 2 inputs -> 4 hidden -> 1 output
    let mut l1 = Linear::new(2, 4);
    let mut l2 = Linear::new(4, 1);

    let lr = 0.5;
    let epochs = 5000;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (x_vec, t_vec) in inputs.iter().zip(targets.iter()) {
            // Prepare input
            let x = Variable::new(Tensor::new(x_vec.clone(), [1, 2]).unwrap());
            let t = Variable::new(Tensor::new(t_vec.clone(), [1, 1]).unwrap());

            // Forward
            let h1 = l1.forward(&x);
            let h1_act = sigmoid(&h1);

            let h2 = l2.forward(&h1_act);
            let pred = sigmoid(&h2); // Sigmoid output for 0/1

            // Loss = (pred - target)^2
            let neg_one = Variable::new(Tensor::new(vec![-1.0], [1, 1]).unwrap());
            let neg_target = t * neg_one;
            let diff = pred.clone() + neg_target;
            let loss = diff.clone() * diff.clone();

            total_loss += loss.data.data()[0];

            // Backward
            loss.backward();

            // Update weights
            update_param(&mut l1.weight, lr);
            update_param(&mut l1.bias, lr);
            update_param(&mut l2.weight, lr);
            update_param(&mut l2.bias, lr);

            // Zero gradients
            zero_grad(&l1.weight);
            zero_grad(&l1.bias);
            zero_grad(&l2.weight);
            zero_grad(&l2.bias);
        }

        if epoch % 500 == 0 {
            println!("Epoch {}: Loss = {}", epoch, total_loss);
        }
    }

    // Verify predictions
    println!("Predictions:");
    for (x_vec, t_vec) in inputs.iter().zip(targets.iter()) {
        let x = Variable::new(Tensor::new(x_vec.clone(), [1, 2]).unwrap());
        let h1 = l1.forward(&x);
        let h1_act = sigmoid(&h1);
        let h2 = l2.forward(&h1_act);
        let pred = sigmoid(&h2);

        let val = pred.data.data()[0];
        println!("Input: {:?}, Target: {:?}, Pred: {}", x_vec, t_vec, val);

        let target = t_vec[0];
        assert!((val - target).abs() < 0.4);
    }
}

fn update_param<const RANK: usize>(var: &mut Variable<f32, RANK>, lr: f32) {
    let data = var.data.data_mut();
    if let Some(grad) = var.grad.borrow().as_ref() {
        let grad_data = grad.data();
        for (w, g) in data.iter_mut().zip(grad_data.iter()) {
            *w -= lr * g;
        }
    }
}

fn zero_grad<const RANK: usize>(var: &Variable<f32, RANK>) {
    *var.grad.borrow_mut() = None;
}
