#![cfg(feature = "models")]

use xla_rs::models::gpt2::{GPT2Block, GPT2Config, GPT2LMHeadModel, GPT2MLP, GPT2Model};
use xla_rs::models::traits::CausalLM;
use xla_rs::nn::transformer::attention::MultiHeadAttention;
use xla_rs::nn::{Embedding, LayerNorm, Linear};
use xla_rs::tensor::Tensor;

fn create_dummy_gpt2(config: &GPT2Config) -> GPT2Model<f32> {
    let mut h = Vec::new();

    for _ in 0..config.n_layer {
        let dim = config.n_embd;
        let head_dim = dim / config.n_head;

        let attn = MultiHeadAttention::new(
            dim,
            config.n_head,
            config.n_head,
            head_dim,
            Linear::new(Tensor::ones([dim, dim]), None),
            Linear::new(Tensor::ones([dim, dim]), None),
            Linear::new(Tensor::ones([dim, dim]), None),
            Linear::new(Tensor::ones([dim, dim]), None),
        );

        let mlp = GPT2MLP {
            c_fc: Linear::new(Tensor::ones([4 * dim, dim]), None),
            c_proj: Linear::new(Tensor::ones([dim, 4 * dim]), None),
        };

        h.push(GPT2Block {
            ln_1: LayerNorm::new(
                Tensor::ones([dim]),
                Tensor::zeros([dim]),
                config.layer_norm_epsilon,
            ),
            attn,
            ln_2: LayerNorm::new(
                Tensor::ones([dim]),
                Tensor::zeros([dim]),
                config.layer_norm_epsilon,
            ),
            mlp,
        });
    }

    GPT2Model {
        wte: Embedding::new(Tensor::zeros([config.vocab_size, config.n_embd])),
        wpe: Embedding::new(Tensor::zeros([config.n_positions, config.n_embd])),
        h,
        ln_f: LayerNorm::new(
            Tensor::ones([config.n_embd]),
            Tensor::zeros([config.n_embd]),
            config.layer_norm_epsilon,
        ),
    }
}

#[test]
fn test_gpt2_forward() {
    let config = GPT2Config {
        vocab_size: 100,
        n_positions: 20,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        layer_norm_epsilon: 1e-5,
    };

    let model = create_dummy_gpt2(&config);

    // Input: [Batch=1, Seq=5]
    let input_ids = Tensor::new(vec![0, 1, 2, 3, 4], [1, 5]).unwrap();

    let output = model.forward(&input_ids, None).unwrap();

    assert_eq!(output.shape(), &[1, 5, 32]);
}

#[test]
fn test_gpt2_lm_head() {
    let config = GPT2Config {
        vocab_size: 100,
        n_positions: 20,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        layer_norm_epsilon: 1e-5,
    };

    let model = create_dummy_gpt2(&config);
    let lm_head = Linear::new(Tensor::zeros([config.vocab_size, config.n_embd]), None);

    let gpt2 = GPT2LMHeadModel {
        transformer: model,
        lm_head,
        config: config.clone(),
    };

    let input_ids = Tensor::new(vec![0, 1], [1, 2]).unwrap();
    let logits = gpt2.forward(&input_ids).unwrap();

    assert_eq!(logits.shape(), &[1, 2, 100]);
}

#[test]
fn test_gpt2_generate() {
    let config = GPT2Config {
        vocab_size: 100,
        n_positions: 20,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        layer_norm_epsilon: 1e-5,
    };

    let model = create_dummy_gpt2(&config);
    let lm_head = Linear::new(Tensor::zeros([config.vocab_size, config.n_embd]), None);

    let gpt2 = GPT2LMHeadModel {
        transformer: model,
        lm_head,
        config: config.clone(),
    };

    let input_ids = Tensor::new(vec![0, 1], [1, 2]).unwrap();
    let generated = gpt2.generate(&input_ids, 3).unwrap();

    assert_eq!(generated.shape(), &[1, 5]);
}
