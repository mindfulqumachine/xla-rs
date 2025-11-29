use xla_rs::nn::Linear;
use xla_rs::nn::transformer::attention::MultiHeadAttention;
use xla_rs::nn::transformer::rope::precompute_freqs_cis;
use xla_rs::tensor::Tensor;

#[test]
fn test_attention_forward() {
    let dim = 16;
    let num_heads = 4;
    let num_kv_heads = 4; // Standard MHA for simplicity
    let head_dim = 4;

    // Create dummy projections (Identity-like would be ideal, but random/ones is fine for shape check)
    let q_proj = Linear::new(Tensor::<f32, 2>::ones([dim, dim]), None);
    let k_proj = Linear::new(Tensor::<f32, 2>::ones([dim, dim]), None);
    let v_proj = Linear::new(Tensor::<f32, 2>::ones([dim, dim]), None);
    let o_proj = Linear::new(Tensor::<f32, 2>::ones([dim, dim]), None);

    let attn = MultiHeadAttention::new(
        dim,
        num_heads,
        num_kv_heads,
        head_dim,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
    );

    // Input: [Batch=1, Seq=2, Dim=16]
    let x = Tensor::<f32, 3>::ones([1, 2, dim]);

    // RoPE freqs
    let (cos, sin) = precompute_freqs_cis(head_dim, 10, 10000.0);

    let output = attn.forward(&x, &cos, &sin, None).unwrap();

    assert_eq!(output.shape(), &[1, 2, dim]);
}
