use criterion::{Criterion, black_box, criterion_group, criterion_main};
use xla_rs::nn::Linear;
use xla_rs::nn::transformer::attention::MultiHeadAttention;
use xla_rs::nn::transformer::attention_optimized::OptimizedMultiHeadAttention;
use xla_rs::tensor::Tensor;

fn benchmark_attention(c: &mut Criterion) {
    let batch = 1;
    let seq_len = 64; // Smaller sequence length to keep benchmark fast
    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = 32;
    let model_dim = num_heads * head_dim;

    // Setup tensors
    // Use unwrap() freely as this is a benchmark setup
    let x = Tensor::new(
        vec![0.5; batch * seq_len * model_dim],
        [batch, seq_len, model_dim],
    )
    .unwrap();
    let freqs_cos =
        Tensor::new(vec![1.0; seq_len * head_dim / 2], [seq_len, head_dim / 2]).unwrap();
    let freqs_sin =
        Tensor::new(vec![0.0; seq_len * head_dim / 2], [seq_len, head_dim / 2]).unwrap();

    // Setup weights (dummy)
    let w_q = Tensor::new(vec![0.1; model_dim * model_dim], [model_dim, model_dim]).unwrap();
    let w_k = Tensor::new(vec![0.1; model_dim * model_dim], [model_dim, model_dim]).unwrap();
    let w_v = Tensor::new(vec![0.1; model_dim * model_dim], [model_dim, model_dim]).unwrap();
    let w_o = Tensor::new(vec![0.1; model_dim * model_dim], [model_dim, model_dim]).unwrap();

    // Pedantic Attention
    let pedantic_attn = MultiHeadAttention::new(
        model_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        Linear::new(w_q.clone(), None),
        Linear::new(w_k.clone(), None),
        Linear::new(w_v.clone(), None),
        Linear::new(w_o.clone(), None),
    );

    // Optimized Attention
    // Linear is not Clone, so we create new ones.
    // We can reuse the weight tensors though as they are Clone (or we clone them).
    let optimized_attn = OptimizedMultiHeadAttention::new(
        num_heads,
        num_kv_heads,
        head_dim,
        Linear::new(w_q.clone(), None),
        Linear::new(w_k.clone(), None),
        Linear::new(w_v.clone(), None),
        Linear::new(w_o.clone(), None),
    )
    .unwrap();

    let mut group = c.benchmark_group("attention");

    group.bench_function("pedantic", |b| {
        b.iter(|| {
            pedantic_attn
                .forward(
                    black_box(&x),
                    black_box(&freqs_cos),
                    black_box(&freqs_sin),
                    black_box(None),
                )
                .unwrap()
        })
    });

    group.bench_function("optimized", |b| {
        b.iter(|| {
            let mut kv_cache = None;
            optimized_attn
                .forward(
                    black_box(&x),
                    black_box(&freqs_cos),
                    black_box(&freqs_sin),
                    black_box(&mut kv_cache),
                    black_box(0),
                )
                .unwrap()
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_attention);
criterion_main!(benches);
