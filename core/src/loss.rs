use crate::autograd::Variable;
use crate::tensor::{Tensor, TensorElem};

/// Computes the Direct Preference Optimization (DPO) loss.
///
/// # What is DPO?
///
/// DPO is a stable and efficient method for fine-tuning language models to align with human preferences.
/// Unlike RLHF (Reinforcement Learning from Human Feedback), which requires training a separate reward model
/// and using PPO, DPO optimizes the policy directly using a closed-form loss function.
///
/// # The Math
///
/// The DPO loss is derived from the optimal solution to the KL-constrained reward maximization problem:
///
/// $$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right] $$
///
/// Where:
/// - $y_w$: Winning (chosen) response.
/// - $y_l$: Losing (rejected) response.
/// - $\pi_{\text{ref}}$: Reference model (usually the SFT model, frozen).
/// - $\beta$: Temperature parameter (controls deviation from reference).
///
/// # Arguments
///
/// * `policy_chosen_logprobs`: Log probs of chosen response from policy. Shape: `[Batch, 1]`.
/// * `policy_rejected_logprobs`: Log probs of rejected response from policy. Shape: `[Batch, 1]`.
/// * `ref_chosen_logprobs`: Log probs of chosen response from reference. Shape: `[Batch, 1]`.
/// * `ref_rejected_logprobs`: Log probs of rejected response from reference. Shape: `[Batch, 1]`.
/// * `beta`: Strength of the KL constraint (typically 0.1).
///
/// # Returns
///
/// A scalar `Variable` representing the mean loss.
pub fn dpo_loss<T: TensorElem + 'static, const RANK: usize>(
    policy_chosen_logprobs: &Variable<T, RANK>,
    policy_rejected_logprobs: &Variable<T, RANK>,
    ref_chosen_logprobs: &Variable<T, RANK>,
    ref_rejected_logprobs: &Variable<T, RANK>,
    beta: T,
) -> Variable<T, RANK> {
    // 1. Calculate log probability ratios
    // log(pi(yw)/ref(yw)) = log(pi(yw)) - log(ref(yw))
    let chosen_logratios = policy_chosen_logprobs.clone() - ref_chosen_logprobs.clone();
    let rejected_logratios = policy_rejected_logprobs.clone() - ref_rejected_logprobs.clone();

    // 2. Calculate logits for the sigmoid
    // logits = beta * (chosen_ratio - rejected_ratio)
    let logits = chosen_logratios - rejected_logratios;

    // Scale by beta. Variable doesn't have scalar mul yet, so we broadcast.
    // Assuming RANK=2 [Batch, 1]
    let shape = *logits.data.shape();
    let beta_tensor = Variable::new(Tensor::new(vec![beta; logits.data.size()], shape).unwrap());
    let scaled_logits = logits * beta_tensor;

    // 3. Compute -log(sigmoid(logits))
    // -log(1 / (1 + e^-x)) = - ( log(1) - log(1 + e^-x) ) = log(1 + e^-x)
    // This formulation is more numerically stable than explicit sigmoid.

    // zero - logits = -logits
    let zero = Variable::new(Tensor::zeros(shape));
    let neg_logits = zero - scaled_logits;

    // exp(-logits)
    let exp_neg_logits = neg_logits.exp();

    // 1 + exp(-logits)
    let one = Variable::new(Tensor::ones(shape));
    let one_plus_exp = one + exp_neg_logits;

    // log(1 + exp(-logits))
    one_plus_exp.log()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_dpo_loss() {
        // Create dummy variables
        // policy_chosen: [0.0], policy_rejected: [0.0]
        // ref_chosen: [0.0], ref_rejected: [0.0]
        // log_ratio_chosen = 0 - 0 = 0
        // log_ratio_rejected = 0 - 0 = 0
        // logits = 0 - 0 = 0
        // loss = -log(sigmoid(0)) = -log(0.5) = 0.6931

        let policy_chosen = Variable::new(Tensor::new(vec![0.0], [1]).unwrap());
        let policy_rejected = Variable::new(Tensor::new(vec![0.0], [1]).unwrap());
        let ref_chosen = Variable::new(Tensor::new(vec![0.0], [1]).unwrap());
        let ref_rejected = Variable::new(Tensor::new(vec![0.0], [1]).unwrap());

        let loss = dpo_loss(
            &policy_chosen,
            &policy_rejected,
            &ref_chosen,
            &ref_rejected,
            1.0,
        );

        let val = loss.data.data()[0];
        assert!((val - 0.693147f64).abs() < 1e-4);
    }

    #[test]
    fn test_dpo_loss_preference() {
        // policy prefers chosen: chosen=1.0, rejected=0.0
        // ref neutral: chosen=0.0, rejected=0.0
        // log_ratio_chosen = 1 - 0 = 1
        // log_ratio_rejected = 0 - 0 = 0
        // logits = 1 - 0 = 1
        // loss = -log(sigmoid(1)) = -log(0.731) = 0.313

        let policy_chosen = Variable::new(Tensor::new(vec![1.0], [1]).unwrap());
        let policy_rejected = Variable::new(Tensor::new(vec![0.0], [1]).unwrap());
        let ref_chosen = Variable::new(Tensor::new(vec![0.0], [1]).unwrap());
        let ref_rejected = Variable::new(Tensor::new(vec![0.0], [1]).unwrap());

        let loss = dpo_loss(
            &policy_chosen,
            &policy_rejected,
            &ref_chosen,
            &ref_rejected,
            1.0,
        );

        let val = loss.data.data()[0];
        assert!((val - 0.31326f64).abs() < 1e-4);
    }
}
