use xla::{ElementType, Result, XlaBuilder, XlaComputation, XlaOp};

/// Returns the replica ID as a scalar U32 tensor.
pub fn replica_id(builder: &XlaBuilder) -> Result<XlaOp> {
    builder.replica_id()
}

/// Performs an All-Reduce sum across all replicas.
pub fn all_reduce_sum(input: &XlaOp) -> Result<XlaOp> {
    let builder = input.builder();
    let shape = input.shape()?;
    let element_type = shape.element_type();

    // Create the reduction computation (add)
    let reduction_computation = XlaComputation::add(
        &builder,
        element_type,
        &[],
        &builder
            .constant_scalar(0f32)
            .convert_element_type(element_type)?,
    )?;

    builder.all_reduce(
        input,
        &reduction_computation,
        &[], // Empty list means all replicas
        None,
        None,
    )
}

/// Performs an All-Gather across all replicas along the specified dimension.
pub fn all_gather(input: &XlaOp, dim: i64) -> Result<XlaOp> {
    let builder = input.builder();
    builder.all_gather(
        input,
        dim,
        0,    // shard_count (0 means all replicas?) - need to verify XLA API
        &[],  // replica_groups
        None, // channel_id
        None, // layout
    )
}
