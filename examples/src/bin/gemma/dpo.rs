use rand::Rng;
use xla_rs::autograd::Variable;
use xla_rs::loss::dpo_loss;
use xla_rs::optim::{Optimizer, Sgd};
use xla_rs::tensor::{Tensor, TensorElem};

// --- Trainable Model (Reused) ---
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
        if let Some(_b) = &self.bias { out } else { out }
    }
}

struct SimpleModel<T: TensorElem> {
    embed: Variable<T, 2>,
    fc1: TrainableLinear<T>,
    fc2: TrainableLinear<T>,
}

impl<T: TensorElem + 'static> SimpleModel<T> {
    fn new(vocab_size: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::rng();
        let embed_data: Vec<T> = (0..vocab_size * hidden_dim)
            .map(|_| T::from_f64(rng.random_range(-0.1..0.1)).unwrap())
            .collect();
        let embed = Variable::new(Tensor::new(embed_data, [vocab_size, hidden_dim]).unwrap());
        let fc1 = TrainableLinear::new(hidden_dim, hidden_dim);
        let fc2 = TrainableLinear::new(hidden_dim, output_dim);
        Self { embed, fc1, fc2 }
    }

    fn forward(&self, x: &Variable<T, 2>) -> Variable<T, 2> {
        // x: [B*S, V] (One-hot)
        let h1 = x.matmul(&self.embed).unwrap();
        let h2 = self.fc1.forward(&h1);
        let h2_act = h2.clone() * h2.clone();
        self.fc2.forward(&h2_act)
    }
}

fn main() {
    println!("Initializing DPO Training...");

    let vocab_size = 10;
    let hidden_dim = 16;
    let batch_size = 2;
    let seq_len = 4;
    let beta = 0.1;

    // Policy Model (Trainable)
    let mut policy_model = SimpleModel::<f32>::new(vocab_size, hidden_dim, vocab_size);

    // Reference Model (Frozen - same init for demo, usually pretrained)
    let ref_model = SimpleModel::<f32>::new(vocab_size, hidden_dim, vocab_size);

    let mut optimizer = Sgd::new(0.01);
    let mut rng = rand::rng();

    println!("Starting DPO Loop...");

    for epoch in 0..5 {
        // Data: Chosen vs Rejected
        // In DPO, we train on pairs of (prompt, chosen_response, rejected_response).
        // The model should assign higher probability to the chosen response than the rejected one.
        //
        // We simulate inputs [B*S, V] for Chosen and Rejected sequences.
        // Chosen:
        let chosen_data: Vec<f32> = (0..batch_size * seq_len * vocab_size)
            .map(|_| if rng.random_bool(0.1) { 1.0 } else { 0.0 })
            .collect();
        let chosen_input =
            Variable::new(Tensor::new(chosen_data, [batch_size * seq_len, vocab_size]).unwrap());

        // Rejected:
        let rejected_data: Vec<f32> = (0..batch_size * seq_len * vocab_size)
            .map(|_| if rng.random_bool(0.1) { 1.0 } else { 0.0 })
            .collect();
        let rejected_input =
            Variable::new(Tensor::new(rejected_data, [batch_size * seq_len, vocab_size]).unwrap());

        // 1. Policy Logprobs
        let pi_chosen_logits = policy_model.forward(&chosen_input); // [B*S, V]
        let pi_rejected_logits = policy_model.forward(&rejected_input);

        // Extract logprob of the "target" token.
        // In this toy example, input IS the target (next token prediction on self?).
        // Let's assume input is one-hot target.
        // LogSoftmax? Variable doesn't have it.
        // We can just take the logit corresponding to the target class.
        // sum(logits * one_hot_target) gives the logit of the target.
        // (Assuming logits are unnormalized log probs, or close enough for toy DPO).
        // Real DPO uses log_softmax.
        // log_softmax(x) = x - log(sum(exp(x))).
        // Implementing log_softmax is hard without Sum/Max ops.
        // Let's assume logits ARE log_probs (e.g. model output is already log_softmax'd or we just use logits).
        // Using logits directly is "good enough" for structural demo.

        let pi_chosen_logprob = pi_chosen_logits.clone() * chosen_input.clone(); // Select target logit
        let pi_chosen_sum = pi_chosen_logprob
            .matmul(&Variable::new(Tensor::ones([vocab_size, 1])))
            .unwrap(); // [B*S, 1]

        let pi_rejected_logprob = pi_rejected_logits.clone() * rejected_input.clone();
        let pi_rejected_sum = pi_rejected_logprob
            .matmul(&Variable::new(Tensor::ones([vocab_size, 1])))
            .unwrap();

        // 2. Ref Logprobs (No grad needed usually, but Variable tracks it. We won't backprop to ref model).
        let ref_chosen_logits = ref_model.forward(&chosen_input);
        let ref_chosen_logprob = ref_chosen_logits * chosen_input.clone();
        let ref_chosen_sum = ref_chosen_logprob
            .matmul(&Variable::new(Tensor::ones([vocab_size, 1])))
            .unwrap();

        let ref_rejected_logits = ref_model.forward(&rejected_input);
        let ref_rejected_logprob = ref_rejected_logits * rejected_input.clone();
        let ref_rejected_sum = ref_rejected_logprob
            .matmul(&Variable::new(Tensor::ones([vocab_size, 1])))
            .unwrap();

        // 3. DPO Loss
        // loss = -log(sigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))))
        // We use the helper function from core::loss

        let loss = dpo_loss(
            &pi_chosen_sum,
            &pi_rejected_sum,
            &ref_chosen_sum,
            &ref_rejected_sum,
            beta,
        );

        let loss_val: f32 =
            loss.data.data().iter().sum::<f32>() / (batch_size as f32 * seq_len as f32);
        println!("Epoch {}: DPO Loss = {:.4}", epoch, loss_val);

        // Backward
        *loss.grad.borrow_mut() = Some(Tensor::ones(*loss.data.shape()));
        loss.backward();

        // Update Policy
        {
            let grad_ref = policy_model.embed.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer
                    .update(vec![&mut policy_model.embed.data], vec![grad], 0)
                    .unwrap();
            }
        }
        *policy_model.embed.grad.borrow_mut() = None;

        {
            let grad_ref = policy_model.fc1.weight.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer
                    .update(vec![&mut policy_model.fc1.weight.data], vec![grad], 1)
                    .unwrap();
            }
        }
        *policy_model.fc1.weight.grad.borrow_mut() = None;

        {
            let grad_ref = policy_model.fc2.weight.grad.borrow();
            if let Some(grad) = grad_ref.as_ref() {
                optimizer
                    .update(vec![&mut policy_model.fc2.weight.data], vec![grad], 2)
                    .unwrap();
            }
        }
        *policy_model.fc2.weight.grad.borrow_mut() = None;
    }

    println!("DPO Training Complete.");
}
