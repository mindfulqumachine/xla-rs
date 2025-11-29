#![cfg(feature = "models")]

use xla_rs::models::gemma::{GemmaBlock, GemmaConfig, GemmaModel, MLP};
use xla_rs::nn::transformer::attention::MultiHeadAttention;
use xla_rs::nn::transformer::rope::precompute_freqs_cis;
use xla_rs::nn::{Linear, RMSNorm};
use xla_rs::tensor::Tensor;

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
