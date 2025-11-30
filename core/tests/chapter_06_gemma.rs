#![cfg(feature = "models")]

use xla_rs::models::gemma::{GemmaBlock, GemmaConfig, GemmaForCausalLM, GemmaModel, MLP};
use xla_rs::models::traits::CausalLM;
use xla_rs::nn::transformer::attention::MultiHeadAttention;
use xla_rs::nn::transformer::rope::precompute_freqs_cis;
use xla_rs::nn::{Embedding, Linear, RMSNorm};
use xla_rs::tensor::{Tensor, TensorElem};

fn create_dummy_model(config: &GemmaConfig) -> GemmaModel<f32> {
    let mut layers = Vec::new();

    for _ in 0..config.num_hidden_layers {
        let dim = config.hidden_size;

        let kv_dim = config.num_key_value_heads * config.head_dim;
        let attn = MultiHeadAttention::new(
            dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            Linear::new(Tensor::ones([dim, dim]), None),
            Linear::new(Tensor::ones([kv_dim, dim]), None),
            Linear::new(Tensor::ones([kv_dim, dim]), None),
            Linear::new(Tensor::ones([dim, dim]), None),
        );

        let mlp = MLP {
            gate_proj: Linear::new(Tensor::ones([config.intermediate_size, dim]), None),
            up_proj: Linear::new(Tensor::ones([config.intermediate_size, dim]), None),
            down_proj: Linear::new(Tensor::ones([dim, config.intermediate_size]), None),
        };

        layers.push(GemmaBlock {
            self_attn: attn,
            mlp,
            input_layernorm: RMSNorm::new(Tensor::ones([dim]), config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(Tensor::ones([dim]), config.rms_norm_eps),
        });
    }

    GemmaModel {
        layers,
        norm: RMSNorm::new(Tensor::ones([config.hidden_size]), config.rms_norm_eps),
    }
}

#[test]
fn test_gemma_forward() {
    let config = GemmaConfig::tiny_test();
    let model = create_dummy_model(&config);

    // Input: [Batch=1, Seq=2, Dim=64]
    let x = Tensor::<f32, 3>::ones([1, 2, config.hidden_size]);

    // RoPE
    let (cos, sin) = precompute_freqs_cis(config.head_dim, 10, 10000.0);

    // Step 1: Norm
    let norm_x = model.layers[0].input_layernorm.forward(&x).unwrap();
    assert_eq!(norm_x.shape(), &[1, 2, 64]);

    // Step 2: Attn
    let attn_out = model.layers[0]
        .self_attn
        .forward(&norm_x, &cos, &sin, None)
        .unwrap();
    assert_eq!(attn_out.shape(), &[1, 2, 64]);

    // Step 3: Residual
    let x2 = (&x + &attn_out).unwrap();

    // Step 4: Post Norm
    let norm_x2 = model.layers[0]
        .post_attention_layernorm
        .forward(&x2)
        .unwrap();
    assert_eq!(norm_x2.shape(), &[1, 2, 64]);

    // Step 5: MLP
    let mlp_out = model.layers[0].mlp.forward(&norm_x2).unwrap();
    assert_eq!(mlp_out.shape(), &[1, 2, 64]);

    // Full forward
    let output = model.forward(&x, &cos, &sin, None).unwrap();
    assert_eq!(output.shape(), &[1, 2, config.hidden_size]);
}

#[test]
fn test_gemma_generate() {
    // Use tiny config
    let config = GemmaConfig::tiny_test();
    let hidden_dim = config.hidden_size;
    let vocab_size = config.vocab_size;

    // Create dummy weights
    let embed_w = Tensor::zeros([vocab_size, hidden_dim]);
    let embed = Embedding::new(embed_w);

    let lm_head_w = Tensor::zeros([vocab_size, hidden_dim]);
    let lm_head = Linear::new(lm_head_w, None);

    let norm_w = Tensor::ones([hidden_dim]);
    let norm = RMSNorm::new(norm_w, config.rms_norm_eps);

    // Create one block (identity-ish)
    let q_proj = Linear::new(Tensor::zeros([hidden_dim, hidden_dim]), None);
    let k_proj = Linear::new(
        Tensor::zeros([config.num_key_value_heads * config.head_dim, hidden_dim]),
        None,
    );
    let v_proj = Linear::new(
        Tensor::zeros([config.num_key_value_heads * config.head_dim, hidden_dim]),
        None,
    );
    let o_proj = Linear::new(Tensor::zeros([hidden_dim, hidden_dim]), None);

    let attn = MultiHeadAttention::new(
        hidden_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
    );

    let gate_proj = Linear::new(Tensor::zeros([config.intermediate_size, hidden_dim]), None);
    let up_proj = Linear::new(Tensor::zeros([config.intermediate_size, hidden_dim]), None);
    let down_proj = Linear::new(Tensor::zeros([hidden_dim, config.intermediate_size]), None);
    let mlp = MLP {
        gate_proj,
        up_proj,
        down_proj,
    };

    let block = GemmaBlock {
        self_attn: attn,
        mlp,
        input_layernorm: RMSNorm::new(Tensor::ones([hidden_dim]), config.rms_norm_eps),
        post_attention_layernorm: RMSNorm::new(Tensor::ones([hidden_dim]), config.rms_norm_eps),
    };

    let model = GemmaModel {
        layers: vec![block],
        norm,
    };

    let gemma = GemmaForCausalLM {
        model,
        lm_head,
        embed_tokens: embed,
        config: config.clone(),
    };

    // Input: [Batch=1, Seq=2]
    let input_ids = Tensor::new(vec![0, 1], [1, 2]).unwrap();

    // Generate 3 tokens
    let output_ids = gemma.generate(&input_ids, 3).unwrap();

    // Output should be [1, 2 + 3] = [1, 5]
    assert_eq!(output_ids.shape(), &[1, 5]);

    // Since weights are zero, logits are zero (or constant).
    // Argmax of zeros is 0.
    // So generated tokens should be 0.
    let data = output_ids.data();
    assert_eq!(data[2], 0);
    assert_eq!(data[3], 0);
    assert_eq!(data[4], 0);
}
