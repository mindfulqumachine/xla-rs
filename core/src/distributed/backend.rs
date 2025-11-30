use crate::tensor::{Device, Result, Tensor, TensorElem};

/// Abstraction for a distributed communication backend.
///
/// This trait allows us to write distributed code (like Tensor Parallelism) once, and run it on:
/// - **XLA/NCCL**: Optimized for GPU clusters (Production).
/// - **CPU/Ring**: Educational implementation using channels (Learning/Debugging).
///
/// # 📡 Collective Operations
///
/// Collective operations involve *all* processes (ranks) in the distributed group.
///
/// ## 1. All-Reduce
///
/// Combines data from all ranks and distributes the result back to all ranks.
///
/// **Example (Sum)**:
/// ```text
/// Rank 0: [1]    Rank 1: [2]    Rank 2: [3]
///       \           |           /
///        \          |          /
///         -----> [1+2+3] <-----
///                   |
///       /           |           \
///      /            |            \
/// Rank 0: [6]    Rank 1: [6]    Rank 2: [6]
/// ```
///
/// ## 2. All-Gather
///
/// Concatenates data from all ranks and distributes the full data to all ranks.
///
/// **Example**:
/// ```text
/// Rank 0: [A]    Rank 1: [B]    Rank 2: [C]
///       \           |           /
///        \          |          /
///         -----> [A,B,C] <-----
///                   |
///       /           |           \
///      /            |            \
/// Rank 0: [ABC]  Rank 1: [ABC]  Rank 2: [ABC]
/// ```
pub trait CollectiveBackend: Send + Sync {
    /// Returns the rank (ID) of the current process/thread.
    ///
    /// - Range: `0` to `world_size - 1`.
    /// - Rank 0 is often the "master" that handles logging or saving checkpoints.
    fn rank(&self) -> usize;

    /// Returns the total number of processes/threads (world size).
    fn world_size(&self) -> usize;

    /// Performs an All-Reduce sum on the given tensor.
    ///
    /// # Arguments
    /// * `tensor` - The input tensor. Must be on a device compatible with the backend.
    ///
    /// # Returns
    /// A new tensor containing the sum of inputs from all ranks.
    fn all_reduce_sum<T: TensorElem, D: Device + 'static>(
        &self,
        tensor: &Tensor<T, 2, D>,
    ) -> Result<Tensor<T, 2, D>>;

    /// Performs an All-Gather on the given tensor along a dimension.
    ///
    /// # Arguments
    /// * `tensor` - The input tensor.
    /// * `dim` - The dimension to gather along (concatenate).
    ///
    /// # Returns
    /// A new tensor containing the concatenated inputs from all ranks.
    fn all_gather<T: TensorElem, D: Device + 'static>(
        &self,
        tensor: &Tensor<T, 2, D>,
        dim: usize,
    ) -> Result<Tensor<T, 2, D>>;
}
