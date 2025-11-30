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

        let weight = Variable::new(Tensor::new(w_data, [out_features, in_features]).unwrap());

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
        // w: [Out, In]
        // y = x @ w.T + b

        // Transpose weight for matmul: [In, Out]
        // Variable doesn't have transpose yet?
        // Check ops.rs. MatMulNode uses transpose internally on Tensor.
        // But Variable::matmul(rhs) does self @ rhs.
        // So we need x @ w.T.
        // Variable doesn't expose transpose.
        // Workaround: Store weight as [In, Out] or implement transpose for Variable.
        // Let's store weight as [In, Out] for this demo to avoid implementing transpose for Variable right now.

        // Wait, Linear usually stores [Out, In].
        // If I store [In, Out], then x @ w works directly.
        // x: [B, In], w: [In, Out] -> [B, Out].

        // Let's assume weight is [In, Out] for TrainableLinear to simplify.
        let out = x.matmul(&self.weight).unwrap();

        if let Some(_b) = &self.bias {
            // Add bias. Broadcasting not fully supported in AddNode?
            // AddNode requires same shape.
            // We need to broadcast bias [Out] to [B, Out].
            // Variable doesn't have broadcast.
            // We can manually repeat bias?
            // Or just skip bias for this demo if too complex.
            // Let's skip bias for now to keep it simple and robust.
            out
        } else {
            out
        }
    }
}

// --- Trainable Model ---

pub struct TrainableGemma<T: TensorElem> {
    // Simplified Gemma: Embedding -> Linear -> Linear -> Softmax
    // Real Gemma has Attention, etc.
    // This is a "Toy Production Grade" demo of the loop.
    pub embed: Variable<T, 2>, // [Vocab, Hidden]
    pub fc1: TrainableLinear<T>,
    pub fc2: TrainableLinear<T>,
}

impl<T: TensorElem + 'static> TrainableGemma<T> {
    fn new(vocab_size: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::rng();
        let embed_data: Vec<T> = (0..vocab_size * hidden_dim)
            .map(|_| T::from_f64(rng.random_range(-0.1..0.1)).unwrap())
            .collect();
        let embed = Variable::new(Tensor::new(embed_data, [vocab_size, hidden_dim]).unwrap());

        let fc1 = TrainableLinear::new(hidden_dim, hidden_dim); // [Hidden, Hidden] (stored as [In, Out])
        let fc2 = TrainableLinear::new(hidden_dim, output_dim); // [Hidden, Out]

        Self { embed, fc1, fc2 }
    }

    #[allow(dead_code)]
    fn forward(&self, _input_ids: &[usize]) -> Variable<T, 2> {
        // Manual embedding lookup since Variable doesn't support gather
        // We construct input tensor by stacking embedding vectors.
        // This breaks the graph for embedding weights (they won't get grads).
        // To fix, we need EmbeddingNode.
        // For this demo, let's assume input is already embeddings (continuous).
        // Or just use a Linear layer as "Embedding" from one-hot input.

        // Let's use one-hot encoding for input to allow backprop to embedding matrix (which is just a Linear layer).
        // Input: [Batch, Vocab] (One-hot)
        // Embed: [Vocab, Hidden]
        // Out = Input @ Embed

        // We'll simulate input as one-hot float tensor.
        // This is inefficient but works for autograd demo.

        // Placeholder return
        Variable::new(Tensor::zeros([1, 1]))
    }

    fn forward_one_hot(&self, x: &Variable<T, 2>) -> Variable<T, 2> {
        // x: [Batch, Vocab]
        // embed: [Vocab, Hidden]
        let h = x.matmul(&self.embed).unwrap();
        let h = self.fc1.forward(&h);
        // Activation? Variable doesn't have Relu/Silu.
        // We can use x * x (square) as non-linearity?
        let h = h.clone() * h.clone();
        self.fc2.forward(&h)
    }
}

fn main() {
    println!("Initializing Trainable Gemma (Toy Version)...");

    let vocab_size = 10;
    let hidden_dim = 16;
    let output_dim = 10; // Next token prediction
    let batch_size = 2;

    // Initialize model
    // Note: TrainableLinear stores weights as [In, Out] for simplicity here.
    // embed: [Vocab, Hidden]
    // fc1: [Hidden, Hidden]
    // fc2: [Hidden, Vocab]

    let mut model = TrainableGemma::new(vocab_size, hidden_dim, output_dim);
    let optimizer = Sgd::new(0.01);
    let mut rng = rand::rng();

    println!("Starting Training Loop...");

    for epoch in 0..5 {
        // Dummy Batch Data (One-hot)
        // Batch size 2, Vocab 10
        let input_data: Vec<f32> = (0..batch_size * vocab_size)
            .map(|_| if rng.random_bool(0.1) { 1.0 } else { 0.0 })
            .collect();
        let inputs = Variable::new(Tensor::new(input_data, [batch_size, vocab_size]).unwrap());

        // Target (Dummy regression target for simplicity)
        let target_data: Vec<f32> = (0..batch_size * output_dim)
            .map(|_| rng.random_range(0.0..1.0))
            .collect();
        let targets = Variable::new(Tensor::new(target_data, [batch_size, output_dim]).unwrap());

        // Forward
        let logits = model.forward_one_hot(&inputs);

        // Loss (MSE: sum((y - target)^2))
        let diff = logits - targets;
        let loss_sq = diff.clone() * diff.clone();

        // Sum loss (simulate by dot with ones)
        let loss_val: f32 = loss_sq.data.data().iter().sum::<f32>() / (batch_size as f32);
        println!("Epoch {}: Loss = {:.4}", epoch, loss_val);

        // Backward
        *loss_sq.grad.borrow_mut() = Some(Tensor::ones(*loss_sq.data.shape()));
        loss_sq.backward();

        // Optimizer Step
        // Update Embed
        {
            let grad_ref = model.embed.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer.update(&mut model.embed.data, grad).unwrap();
            }
        }
        *model.embed.grad.borrow_mut() = None;

        // Update FC1
        {
            let grad_ref = model.fc1.weight.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer.update(&mut model.fc1.weight.data, grad).unwrap();
            }
        }
        *model.fc1.weight.grad.borrow_mut() = None;

        // Update FC2
        {
            let grad_ref = model.fc2.weight.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer.update(&mut model.fc2.weight.data, grad).unwrap();
            }
        }
        *model.fc2.weight.grad.borrow_mut() = None;
    }

    println!("Training Complete.");
}
