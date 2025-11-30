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

    let output = attn.forward(&x, Some(&cos), Some(&sin), None).unwrap();

    assert_eq!(output.shape(), &[1, 2, dim]);
}

#[test]
fn test_attention_pronoun_resolution() {
    use xla_rs::nn::transformer::attention_optimized::{KVCache, OptimizedMultiHeadAttention};
    // This test simulates a "real-world" scenario: Pronoun Resolution.
    // We want to see if the model can resolve what "it" refers to in two different contexts.
    //
    // Sentence A: "The trophy would not fit in the suitcase because it was too big."
    // -> "it" refers to "trophy" (because trophies are big).
    //
    // Sentence B: "The trophy would not fit in the suitcase because it was too small."
    // -> "it" refers to "suitcase" (because suitcases are small).
    //
    // In a real Transformer, the embedding of "it" would be "contextualized" by previous layers
    // (Self-Attention and MLP) to absorb information from "big" or "small".
    // Since we are testing a single Attention layer here, we will *manually* simulate this
    // contextualization by crafting the input embedding of "it" to be similar to "trophy"
    // in Case A, and similar to "suitcase" in Case B.

    // let b = 1;
    let num_heads = 1; // Single head for clarity
    let num_kv_heads = 1;
    let head_dim = 8; // Small dimension for easy manual construction
    let hidden_dim = 8;

    // 1. Vocabulary & Mock Embeddings
    // We'll use a simplified vocabulary and assign random-ish vectors.
    // Key words: "trophy", "suitcase", "it"
    let vocab = [
        "The", "trophy", "would", "not", "fit", "in", "the", "suitcase", "because", "it", "was",
        "too", "big", "small", ".",
    ];

    // Helper to create a one-hot vector (deterministic for test)
    let make_vec = |seed: u8| -> Vec<f32> {
        let mut v = vec![0.0; hidden_dim];
        v[seed as usize % hidden_dim] = 1.0;
        v
    };

    let vec_trophy = make_vec(11); // "trophy" vector
    let vec_suitcase = make_vec(22); // "suitcase" vector
    let vec_other = make_vec(33); // generic vector for other words

    // 2. Setup Attention Layer
    // We use Identity projections so that Q, K, V are just the input embeddings.
    // This allows us to directly control the dot products via input embeddings.
    // Q = x, K = x, V = x.
    // Score = x_i @ x_j.T
    let mut identity_data = vec![0.0; hidden_dim * hidden_dim];
    for i in 0..hidden_dim {
        identity_data[i * hidden_dim + i] = 1.0;
    }
    let identity_w = Tensor::<f32, 2>::new(identity_data, [hidden_dim, hidden_dim]).unwrap();
    let q_proj = Linear::new(identity_w.clone(), None);
    let k_proj = Linear::new(identity_w.clone(), None);
    let v_proj = Linear::new(identity_w.clone(), None);
    let o_proj = Linear::new(identity_w.clone(), None);

    let attn = OptimizedMultiHeadAttention::new(
        num_heads,
        num_kv_heads,
        head_dim,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
    )
    .unwrap();

    // 3. Construct Scenarios
    // We only need to simulate the query "it" attending to the past keys.
    // We will assume "it" is the current token we are processing (inference mode).
    // The past tokens (KV cache) are "The trophy ... because".

    // Scenario A: "it" (context: big) -> looks for "trophy"
    // We make "it" embedding close to "trophy" embedding.
    // In dot-product attention, similarity is key.
    // Let's just set query "it" = "trophy" vector.
    let query_a = Tensor::<f32, 3>::new(vec_trophy.clone(), [1, 1, hidden_dim]).unwrap();

    // Scenario B: "it" (context: small) -> looks for "suitcase"
    // Set query "it" = "suitcase" vector.
    let query_b = Tensor::<f32, 3>::new(vec_suitcase.clone(), [1, 1, hidden_dim]).unwrap();

    // Construct KV Cache (Past tokens)
    // Sequence: "The trophy would not fit in the suitcase because"
    // Indices:   0     1      2    3   4   5   6       7         8
    // "trophy" is at index 1.
    // "suitcase" is at index 7.
    let seq_len = 9;
    let mut kv_data = Vec::new();
    for (_, word) in vocab.iter().take(seq_len).enumerate() {
        if *word == "trophy" {
            kv_data.extend_from_slice(&vec_trophy);
        } else if *word == "suitcase" {
            kv_data.extend_from_slice(&vec_suitcase);
        } else {
            kv_data.extend_from_slice(&vec_other);
        }
    }

    // Create K and V tensors (Batch=1, Seq=9, Dim=8)
    // We need to format them for the KV cache update or just pre-fill.
    // The `update` takes [B, H, S, D].
    // Here H=1.
    let k_tensor = Tensor::<f32, 4>::new(kv_data.clone(), [1, 1, seq_len, hidden_dim]).unwrap();
    let v_tensor = Tensor::<f32, 4>::new(kv_data.clone(), [1, 1, seq_len, hidden_dim]).unwrap();

    // Initialize Cache
    let mut cache = Some(KVCache::new(1, 1, 20, hidden_dim));
    cache
        .as_mut()
        .unwrap()
        .update(&k_tensor, &v_tensor, 0)
        .unwrap();

    // Dummy RoPE (Identity)
    let freqs_cos = Tensor::<f32, 2>::ones([1, head_dim / 2]);
    let freqs_sin = Tensor::<f32, 2>::zeros([1, head_dim / 2]);

    // 4. Run & Verify Case A
    println!("\n=== Case A: '...because it was too big' ===");
    println!("Query 'it' (contextualized) is similar to 'trophy'.");
    let (_, weights_a) = attn
        .forward_with_weights(&query_a, &freqs_cos, &freqs_sin, &mut cache, seq_len)
        .unwrap();
    let weights_a = weights_a.unwrap(); // [B, H, S_q, Total_S] -> [1, 1, 1, 9]

    let w_data_a = weights_a.data();
    let score_trophy_a = w_data_a[1]; // Index 1
    let score_suitcase_a = w_data_a[7]; // Index 7

    print_heatmap(&vocab[..seq_len], w_data_a);

    assert!(
        score_trophy_a > score_suitcase_a,
        "In Case A, 'it' should attend more to 'trophy' ({:.4}) than 'suitcase' ({:.4})",
        score_trophy_a,
        score_suitcase_a
    );

    // 5. Run & Verify Case B
    println!("\n=== Case B: '...because it was too small' ===");
    println!("Query 'it' (contextualized) is similar to 'suitcase'.");
    let (_, weights_b) = attn
        .forward_with_weights(&query_b, &freqs_cos, &freqs_sin, &mut cache, seq_len)
        .unwrap();
    let weights_b = weights_b.unwrap();
    let w_data_b = weights_b.data();
    let score_trophy_b = w_data_b[1];
    let score_suitcase_b = w_data_b[7];

    print_heatmap(&vocab[..seq_len], w_data_b);

    assert!(
        score_suitcase_b > score_trophy_b,
        "In Case B, 'it' should attend more to 'suitcase' ({:.4}) than 'trophy' ({:.4})",
        score_suitcase_b,
        score_trophy_b
    );
}

fn print_heatmap(tokens: &[&str], weights: &[f32]) {
    println!("Attention Weights for 'it':");
    for (i, token) in tokens.iter().enumerate() {
        let w = weights[i];
        let bar_len = (w * 20.0).round() as usize;
        let bar = "█".repeat(bar_len);
        println!("{:>10} [{:.4}] {}", token, w, bar);
    }
}
