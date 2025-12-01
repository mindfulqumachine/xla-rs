use rand::Rng;
use xla_rs::autograd::Variable;
use xla_rs::optim::{Optimizer, Sgd};
use xla_rs::tensor::{Tensor, TensorElem};

// --- Trainable LoRA Linear ---

struct TrainableLoraLinear<T: TensorElem> {
    // Frozen weight (Variable but we won't update it)
    frozen_weight: Variable<T, 2>,
    // LoRA weights (Trainable)
    lora_a: Variable<T, 2>, // [In, R]
    lora_b: Variable<T, 2>, // [R, Out]
    scaling: T,
}

impl<T: TensorElem + 'static> TrainableLoraLinear<T> {
    fn new(in_features: usize, out_features: usize, r: usize, alpha: T) -> Self {
        let mut rng = rand::rng();

        // Frozen Weight (Simulate Pretrained)
        let scale = (6.0 / (in_features + out_features) as f64).sqrt();
        let w_data: Vec<T> = (0..in_features * out_features)
            .map(|_| T::from_f64(rng.random_range(-scale..scale)).unwrap())
            .collect();
        let frozen_weight =
            Variable::new(Tensor::new(w_data, [in_features, out_features]).unwrap());

        // LoRA A: Gaussian Init
        let a_data: Vec<T> = (0..in_features * r)
            .map(|_| T::from_f64(rng.random_range(-0.01..0.01)).unwrap())
            .collect();
        let lora_a = Variable::new(Tensor::new(a_data, [in_features, r]).unwrap());

        // LoRA B: Zero Init
        let b_data: Vec<T> = vec![T::zero(); r * out_features];
        let lora_b = Variable::new(Tensor::new(b_data, [r, out_features]).unwrap());

        Self {
            frozen_weight,
            lora_a,
            lora_b,
            scaling: alpha, // alpha / r usually, but we take alpha as scaling factor
        }
    }

    fn forward(&self, x: &Variable<T, 2>) -> Variable<T, 2> {
        // x: [B, In]

        // 1. Frozen Path: x @ W
        let frozen_out = x.matmul(&self.frozen_weight).unwrap();

        // 2. LoRA Path: x @ A @ B * scaling
        let xa = x.matmul(&self.lora_a).unwrap();
        let lora_out = xa.matmul(&self.lora_b).unwrap();

        // Scale (Variable doesn't have scalar mul yet?)
        // We can simulate scalar mul by matmul with diagonal or just map?
        // Variable doesn't support map.
        // We can use a 1x1 tensor broadcast? Variable doesn't broadcast.
        // We can multiply by a Tensor of same shape filled with scaling.
        // Let's create a scaling tensor.

        let scale_tensor = Variable::new(
            Tensor::new(
                vec![self.scaling; lora_out.data.shape().iter().product()],
                *lora_out.data.shape(),
            )
            .unwrap(),
        );
        let scaled_lora = lora_out * scale_tensor;

        // Sum
        frozen_out + scaled_lora
    }
}

fn main() {
    println!("Initializing LoRA Training...");

    let vocab_size = 20;
    let hidden_dim = 16;
    let output_dim = 20;
    let batch_size = 2;
    let r = 4;
    let alpha = 1.0; // scaling

    let mut rng = rand::rng();

    // Model: Embed -> LoraLinear -> LoraLinear -> Output
    // Embed is frozen for this demo (or trainable, but let's focus on LoRA layers)
    let embed_data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|_| rng.random_range(-0.1..0.1))
        .collect();
    let embed = Variable::new(Tensor::new(embed_data, [vocab_size, hidden_dim]).unwrap());

    let mut fc1 = TrainableLoraLinear::<f32>::new(hidden_dim, hidden_dim, r, alpha);
    let mut fc2 = TrainableLoraLinear::<f32>::new(hidden_dim, output_dim, r, alpha);

    let mut optimizer = Sgd::new(0.01);

    println!("Starting LoRA Loop...");

    for epoch in 0..5 {
        // Dummy Data [B*S, V] (Flattened for simplicity as in SFT)
        let seq_len = 4;
        let input_data: Vec<f32> = (0..batch_size * seq_len * vocab_size)
            .map(|_| if rng.random_bool(0.05) { 1.0 } else { 0.0 })
            .collect();
        let inputs =
            Variable::new(Tensor::new(input_data, [batch_size * seq_len, vocab_size]).unwrap());

        let target_data: Vec<f32> = (0..batch_size * seq_len * vocab_size)
            .map(|_| rng.random_range(0.0..1.0))
            .collect();
        let targets =
            Variable::new(Tensor::new(target_data, [batch_size * seq_len, vocab_size]).unwrap());

        // Forward
        let h1 = inputs.matmul(&embed).unwrap();
        let h2 = fc1.forward(&h1);
        let h2_act = h2.clone() * h2.clone();
        let logits = fc2.forward(&h2_act);

        // Loss
        let diff = logits - targets;
        let loss_sq = diff.clone() * diff.clone();

        let loss_val: f32 =
            loss_sq.data.data().iter().sum::<f32>() / (batch_size as f32 * seq_len as f32);
        println!("Epoch {}: LoRA Loss = {:.4}", epoch, loss_val);

        // Backward
        *loss_sq.grad.borrow_mut() = Some(Tensor::ones(*loss_sq.data.shape()));
        loss_sq.backward();

        // Update ONLY LoRA weights
        //
        // Critical Step: We only update lora_a and lora_b.
        // The `frozen_weight` (base model) is NOT updated.
        // This drastically reduces the memory requirement for gradients and optimizer states.
        {
            let grad_ref = fc1.lora_a.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer
                    .update(vec![&mut fc1.lora_a.data], vec![grad], 0)
                    .unwrap();
            }
        }
        *fc1.lora_a.grad.borrow_mut() = None;

        {
            let grad_ref = fc1.lora_b.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer
                    .update(vec![&mut fc1.lora_b.data], vec![grad], 1)
                    .unwrap();
            }
        }
        *fc1.lora_b.grad.borrow_mut() = None;

        {
            let grad_ref = fc2.lora_a.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer
                    .update(vec![&mut fc2.lora_a.data], vec![grad], 2)
                    .unwrap();
            }
        }
        *fc2.lora_a.grad.borrow_mut() = None;

        {
            let grad_ref = fc2.lora_b.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer
                    .update(vec![&mut fc2.lora_b.data], vec![grad], 3)
                    .unwrap();
            }
        }
        *fc2.lora_b.grad.borrow_mut() = None;

        // Frozen weights are NOT updated
    }

    println!("LoRA Training Complete.");
}
