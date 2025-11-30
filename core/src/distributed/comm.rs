use xla::{ElementType, Result, XlaBuilder, XlaComputation, XlaOp};

/// Returns the replica ID as a scalar U32 tensor.
///
/// In XLA, `replica_id` is a special op that returns the unique ID of the device
/// executing the computation.
pub fn replica_id(builder: &XlaBuilder) -> Result<XlaOp> {
    builder.replica_id()
}

/// Performs an All-Reduce sum across all replicas.
///
/// # How it works (XLA/NCCL)
///
/// 1.  **Computation**: We define a "reduction computation" (in this case, addition).
/// 2.  **Topology**: XLA assumes a default topology where all devices participate.
/// 3.  **Execution**: On GPU, this compiles to a NCCL AllReduce kernel.
///
/// # Arguments
/// * `input` - The input tensor (XlaOp).
///
/// # Returns
/// An `XlaOp` representing the sum of inputs from all replicas.
pub fn all_reduce_sum(input: &XlaOp) -> Result<XlaOp> {
    let builder = input.builder();
    let shape = input.shape()?;
    let element_type = shape.element_type();

    // Create the reduction computation (add)
    // This defines "what to do" with two values.
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
///
/// # How it works
///
/// Concatenates the input tensors from all replicas along `dim`.
/// If each replica has a tensor of shape `[H, W]`, and we gather along `dim=0`,
/// the output will be `[H * world_size, W]`.
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
