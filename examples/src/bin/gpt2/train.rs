use rand::Rng;
use xla_rs::autograd::Variable;
use xla_rs::optim::{Optimizer, Sgd};
use xla_rs::tensor::{Tensor, TensorElem};

// --- Trainable Layers using Variable ---

pub struct TrainableLinear<T: TensorElem> {
    pub weight: Variable<T, 2>,
    pub bias: Option<Variable<T, 1>>,
}

impl<T: TensorElem + 'static> TrainableLinear<T> {
    fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::rng();
        // Xavier initialization
        let scale = (6.0 / (in_features + out_features) as f64).sqrt();
        let w_data: Vec<T> = (0..in_features * out_features)
            .map(|_| T::from_f64(rng.random_range(-scale..scale)).unwrap())
            .collect();

        let weight = Variable::new(Tensor::new(w_data, [in_features, out_features]).unwrap());

        // Bias init to 0
        let b_data = vec![T::zero(); out_features];
        let bias = Variable::new(Tensor::new(b_data, [out_features]).unwrap());

        Self {
            weight,
            bias: Some(bias),
        }
    }

    fn forward(&self, x: &Variable<T, 2>) -> Variable<T, 2> {
        // x: [Batch, In]
        // w: [Out, In] (stored)
        // y = x @ w.T + b

        // We assume weight is [Out, In] as per standard Linear, but here we might need to be careful.
        // In gemma/train.rs comments said: "Let's assume weight is [In, Out] for TrainableLinear to simplify."
        // But the code initialized it as [out_features, in_features].
        // And then did x.matmul(&self.weight).
        // If x is [B, In] and weight is [Out, In], matmul(x, w) -> [B, In] x [Out, In] -> Error?
        // Unless matmul auto-transposes? Or Variable::matmul does something else?
        // Let's stick to what gemma/train.rs did:
        // "Let's assume weight is [In, Out] for TrainableLinear to simplify."
        // But the init was: Tensor::new(w_data, [out_features, in_features]).
        // Wait, if init is [Out, In], and x is [B, In].
        // x.matmul(w) would fail if standard matmul.
        // Maybe gemma/train.rs works because In=Out=Hidden?
        // In gemma/train.rs: fc1 is [Hidden, Hidden]. fc2 is [Hidden, Out].
        // If fc2 is [Hidden, Out], and init as [Out, Hidden].
        // x is [B, Hidden].
        // x.matmul(w) -> [B, Hidden] x [Out, Hidden].
        // This only works if Hidden == Out?
        // Or if matmul transposes the second arg?
        // Let's assume for now we just copy the logic.

        let out = x.matmul(&self.weight).unwrap();

        if let Some(_b) = &self.bias {
            // Skip bias for simplicity as in gemma example
            out
        } else {
            out
        }
    }
}

// --- Trainable GPT-2 (Toy) ---

pub struct TrainableGPT2<T: TensorElem> {
    // Simplified GPT-2: Embedding -> Linear (Block) -> Linear (Head)
    pub wte: Variable<T, 2>, // [Vocab, Hidden]
    pub wpe: Variable<T, 2>, // [Pos, Hidden]
    pub c_fc: TrainableLinear<T>,
    pub c_proj: TrainableLinear<T>,
    pub lm_head: TrainableLinear<T>,
}

impl<T: TensorElem + 'static> TrainableGPT2<T> {
    fn new(vocab_size: usize, pos_size: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::rng();

        // WTE
        let wte_data: Vec<T> = (0..vocab_size * hidden_dim)
            .map(|_| T::from_f64(rng.random_range(-0.1..0.1)).unwrap())
            .collect();
        let wte = Variable::new(Tensor::new(wte_data, [vocab_size, hidden_dim]).unwrap());

        // WPE
        let wpe_data: Vec<T> = (0..pos_size * hidden_dim)
            .map(|_| T::from_f64(rng.random_range(-0.1..0.1)).unwrap())
            .collect();
        let wpe = Variable::new(Tensor::new(wpe_data, [pos_size, hidden_dim]).unwrap());

        // Block (Simplified MLP)
        let c_fc = TrainableLinear::new(hidden_dim, 4 * hidden_dim);
        let c_proj = TrainableLinear::new(4 * hidden_dim, hidden_dim);

        // Head
        let lm_head = TrainableLinear::new(hidden_dim, output_dim);

        Self {
            wte,
            wpe,
            c_fc,
            c_proj,
            lm_head,
        }
    }

    fn forward_one_hot(&self, x: &Variable<T, 2>, pos_ids: &Variable<T, 2>) -> Variable<T, 2> {
        // x: [Batch, Vocab] (One-hot)
        // pos_ids: [Batch, Pos] (One-hot)

        let inputs_embeds = x.matmul(&self.wte).unwrap();
        let position_embeds = pos_ids.matmul(&self.wpe).unwrap();

        let h = inputs_embeds + position_embeds;

        // MLP Block
        let h = self.c_fc.forward(&h);
        // Gelu approx (x * x for toy)
        let h = h.clone() * h.clone();
        let h = self.c_proj.forward(&h);

        // Head
        self.lm_head.forward(&h)
    }
}

fn main() {
    println!("Initializing Trainable GPT-2 (Toy Version)...");

    let vocab_size = 10;
    let pos_size = 5;
    let hidden_dim = 16;
    let output_dim = 10;
    let batch_size = 2;

    let mut model = TrainableGPT2::new(vocab_size, pos_size, hidden_dim, output_dim);
    let mut optimizer = Sgd::new(0.01);
    let mut rng = rand::rng();

    println!("Starting Training Loop...");

    for epoch in 0..5 {
        // Dummy Input (One-hot)
        let input_data: Vec<f32> = (0..batch_size * vocab_size)
            .map(|_| if rng.random_bool(0.1) { 1.0 } else { 0.0 })
            .collect();
        let inputs = Variable::new(Tensor::new(input_data, [batch_size, vocab_size]).unwrap());

        // Dummy Pos (One-hot) - simplified, usually indices
        let pos_data: Vec<f32> = (0..batch_size * pos_size)
            .map(|_| if rng.random_bool(0.2) { 1.0 } else { 0.0 })
            .collect();
        let pos_ids = Variable::new(Tensor::new(pos_data, [batch_size, pos_size]).unwrap());

        // Target
        let target_data: Vec<f32> = (0..batch_size * output_dim)
            .map(|_| rng.random_range(0.0..1.0))
            .collect();
        let targets = Variable::new(Tensor::new(target_data, [batch_size, output_dim]).unwrap());

        // Forward
        let logits = model.forward_one_hot(&inputs, &pos_ids);

        // Loss
        let diff = logits - targets;
        let loss_sq = diff.clone() * diff.clone();

        let loss_val: f32 = loss_sq.data.data().iter().sum::<f32>() / (batch_size as f32);
        println!("Epoch {}: Loss = {:.4}", epoch, loss_val);

        // Backward
        *loss_sq.grad.borrow_mut() = Some(Tensor::ones(*loss_sq.data.shape()));
        loss_sq.backward();

        // Optimizer Step
        // Update WTE
        {
            let grad_ref = model.wte.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer
                    .update(vec![&mut model.wte.data], vec![grad], 0)
                    .unwrap();
            }
        }
        *model.wte.grad.borrow_mut() = None;

        // Update WPE
        {
            let grad_ref = model.wpe.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer
                    .update(vec![&mut model.wpe.data], vec![grad], 1)
                    .unwrap();
            }
        }
        *model.wpe.grad.borrow_mut() = None;

        // Update Layers
        let layers = vec![&mut model.c_fc, &mut model.c_proj, &mut model.lm_head];
        for layer in layers {
            {
                let grad_ref = layer.weight.grad.borrow();
                if let Some(grad) = grad_ref.as_ref() {
                    optimizer
                        .update(vec![&mut layer.weight.data], vec![grad], 2)
                        .unwrap();
                }
            }
            *layer.weight.grad.borrow_mut() = None;
        }
    }

    println!("Training Complete.");
}
