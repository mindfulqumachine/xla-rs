#![cfg(feature = "xla-backend")]
use xla::{ElementType, Result, XlaBuilder};
use xla_rs::distributed::linear::{ParallelStrategy, TensorParallelLinear};

#[test]
fn test_tensor_parallel_linear_graph_construction() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = XlaBuilder::new("tp_linear_test");

    // Dimensions
    let batch_size = 4;
    let in_features = 16;
    let out_features = 32;

    // Create input X [batch_size, in_features]
    let x = builder.parameter(0, ElementType::F32, &[batch_size, in_features], "x")?;

    // 1. Test Column Parallel
    // Weight split along output dimension: [in_features, out_features / 2]
    let w_col = builder.constant_r2(
        &vec![0.1f32; (in_features * out_features / 2) as usize],
        in_features,
        out_features / 2,
    )?;
    let b_col =
        builder.constant_r1(&vec![0.1f32; (out_features / 2) as usize], out_features / 2)?;

    let tp_col = TensorParallelLinear::new(w_col, Some(b_col), ParallelStrategy::Column);
    let _y_col = tp_col.forward(&x)?;

    // 2. Test Row Parallel
    // Input for Row Parallel should theoretically be split, but for graph construction we can use full X
    // if we assume X is already the local shard.
    // Weight split along input dimension: [in_features / 2, out_features]
    let w_row = builder.constant_r2(
        &vec![0.1f32; (in_features / 2 * out_features) as usize],
        in_features / 2,
        out_features,
    )?;
    let b_row = builder.constant_r1(&vec![0.1f32; out_features as usize], out_features)?;

    // We need a split input for row parallel: [batch_size, in_features / 2]
    let x_split = builder.parameter(
        1,
        ElementType::F32,
        &[batch_size, in_features / 2],
        "x_split",
    )?;

    let tp_row = TensorParallelLinear::new(w_row, Some(b_row), ParallelStrategy::Row);
    let y_row = tp_row.forward(&x_split)?;

    // Build the computation to ensure XLA accepts the collectives
    let _computation = builder.build(&y_row)?;

    Ok(())
}
