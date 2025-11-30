use crate::autograd::Variable;
use crate::tensor::{Tensor, TensorElem};

/// Computes the Direct Preference Optimization (DPO) loss.
///
/// The DPO loss is defined as:
/// `L_DPO = -log(sigmoid(beta * (log(pi_theta(yw|x)) - log(pi_ref(yw|x)) - (log(pi_theta(yl|x)) - log(pi_ref(yl|x))))))`
///
/// Where:
/// - `pi_theta` is the policy model (being trained).
/// - `pi_ref` is the reference model (frozen).
/// - `yw` is the winning (chosen) response.
/// - `yl` is the losing (rejected) response.
/// - `beta` is a hyperparameter controlling the deviation from the reference model.
///
/// # Arguments
///
/// * `policy_chosen_logprobs` - Log probabilities of the chosen responses from the policy model. Shape: [Batch, 1]
/// * `policy_rejected_logprobs` - Log probabilities of the rejected responses from the policy model. Shape: [Batch, 1]
/// * `ref_chosen_logprobs` - Log probabilities of the chosen responses from the reference model. Shape: [Batch, 1]
/// * `ref_rejected_logprobs` - Log probabilities of the rejected responses from the reference model. Shape: [Batch, 1]
/// * `beta` - Temperature parameter (typically 0.1 to 0.5).
///
/// # Returns
///
/// A scalar Variable representing the mean DPO loss over the batch.
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
