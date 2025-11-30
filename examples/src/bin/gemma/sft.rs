use rand::Rng;
use xla_rs::autograd::Variable;
use xla_rs::optim::{Optimizer, Sgd};
use xla_rs::tensor::{Tensor, TensorElem};

// --- Trainable Model (Reused from train.rs for consistency) ---

struct TrainableLinear<T: TensorElem> {
    weight: Variable<T, 2>,
    bias: Option<Variable<T, 1>>,
}

impl<T: TensorElem + 'static> TrainableLinear<T> {
    fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::rng();
        let scale = (6.0 / (in_features + out_features) as f64).sqrt();
        let w_data: Vec<T> = (0..in_features * out_features)
            .map(|_| T::from_f64(rng.random_range(-scale..scale)).unwrap())
            .collect();

        // Store as [In, Out] for simplified matmul
        let weight = Variable::new(Tensor::new(w_data, [in_features, out_features]).unwrap());

        let b_data = vec![T::zero(); out_features];
        let bias = Variable::new(Tensor::new(b_data, [out_features]).unwrap());

        Self {
            weight,
            bias: Some(bias),
        }
    }

    fn forward(&self, x: &Variable<T, 2>) -> Variable<T, 2> {
        let out = x.matmul(&self.weight).unwrap();
        if let Some(_b) = &self.bias {
            // Bias addition omitted for simplicity as in train.rs
            out
        } else {
            out
        }
    }
}

fn main() {
    println!("Initializing SFT Training...");

    let vocab_size = 20;
    let hidden_dim = 16;
    let output_dim = 20; // Vocab size
    let batch_size = 2;

    let mut rng = rand::rng();

    // Model Params
    let embed_data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|_| rng.random_range(-0.1..0.1))
        .collect();
    let mut embed = Variable::new(Tensor::new(embed_data, [vocab_size, hidden_dim]).unwrap());

    let mut fc1 = TrainableLinear::<f32>::new(hidden_dim, hidden_dim);
    let mut fc2 = TrainableLinear::<f32>::new(hidden_dim, output_dim);

    let mut optimizer = Sgd::new(0.01);

    println!("Starting SFT Loop...");

    for epoch in 0..5 {
        // Mock SFT Batch: (Instruction, Response)
        // We concatenate them: [Instr, Resp]
        // And mask loss for Instr.

        // Input: [Batch, Seq] (One-hot simulation)
        // Let's assume Seq=4. Instr=2, Resp=2.
        // Mask: [0, 0, 1, 1]

        let seq_len = 4;
        let mask_data: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]; // Repeated for batch 2
        let mask = Variable::new(Tensor::new(mask_data, [batch_size, seq_len]).unwrap());

        // Inputs: [Batch, Seq, Vocab] -> Flatten to [Batch*Seq, Vocab]
        let input_data: Vec<f32> = (0..batch_size * seq_len * vocab_size)
            .map(|_| if rng.random_bool(0.05) { 1.0 } else { 0.0 })
            .collect();
        let inputs =
            Variable::new(Tensor::new(input_data, [batch_size * seq_len, vocab_size]).unwrap());

        // Targets: [Batch*Seq, Vocab]
        let target_data: Vec<f32> = (0..batch_size * seq_len * vocab_size)
            .map(|_| rng.random_range(0.0..1.0))
            .collect();
        let targets =
            Variable::new(Tensor::new(target_data, [batch_size * seq_len, vocab_size]).unwrap());

        // Forward (Rank 2)
        // 1. Embed: [B*S, V] @ [V, H] -> [B*S, H]
        let h1 = inputs.matmul(&embed).unwrap();

        // 2. FC1: [B*S, H] @ [H, H] -> [B*S, H]
        let h2 = fc1.forward(&h1);
        let h2_act = h2.clone() * h2.clone(); // Square activation

        // 3. FC2: [B*S, H] @ [H, V] -> [B*S, V]
        let logits = fc2.forward(&h2_act);

        // Loss
        let diff = logits - targets;
        let loss_sq = diff.clone() * diff.clone(); // [B*S, V]

        // Apply Mask
        // Mask: [B, S] -> Flatten to [B*S] -> Reshape to [B*S, 1] for broadcast
        let mask_flat = mask.data.reshape([batch_size * seq_len, 1]).unwrap();
        let mask_var = Variable::new(mask_flat);

        // Sum loss over vocab: [B*S, V] @ Ones [V, 1] -> [B*S, 1]
        let ones_v = Variable::new(Tensor::ones([vocab_size, 1]));
        let loss_per_token = loss_sq.matmul(&ones_v).unwrap(); // [B*S, 1]

        let masked_loss = loss_per_token * mask_var;

        // Mean loss
        let loss_val: f32 = masked_loss.data.data().iter().sum::<f32>() / (batch_size as f32 * 2.0); // Divide by valid tokens
        println!("Epoch {}: SFT Loss = {:.4}", epoch, loss_val);

        // Backward
        *masked_loss.grad.borrow_mut() = Some(Tensor::ones(*masked_loss.data.shape()));
        masked_loss.backward();

        // Update
        {
            let grad_ref = embed.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer.update(0, &mut embed.data, grad).unwrap();
            }
        }
        *embed.grad.borrow_mut() = None;

        {
            let grad_ref = fc1.weight.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer.update(1, &mut fc1.weight.data, grad).unwrap();
            }
        }
        *fc1.weight.grad.borrow_mut() = None;

        {
            let grad_ref = fc2.weight.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer.update(2, &mut fc2.weight.data, grad).unwrap();
            }
        }
        *fc2.weight.grad.borrow_mut() = None;
    }

    println!("SFT Complete.");
}
