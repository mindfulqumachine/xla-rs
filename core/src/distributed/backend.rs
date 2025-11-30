use crate::tensor::{Device, Result, Tensor, TensorElem};

/// Abstraction for a distributed communication backend.
///
/// This trait allows swapping between different communication implementations:
/// - **XLA/NCCL**: Optimized for GPU clusters.
/// - **CPU/Ring**: Educational implementation using channels/TCP.
pub trait CollectiveBackend: Send + Sync {
    /// Returns the rank of the current process/thread.
    fn rank(&self) -> usize;

    /// Returns the total number of processes/threads.
    fn world_size(&self) -> usize;

    /// Performs an All-Reduce sum on the given tensor.
    ///
    /// The input tensor must be on a device compatible with the backend.
    /// For `CpuBackend`, it must be on `Cpu`.
    /// For `XlaBackend`, it must be on `XlaDevice`.
    fn all_reduce_sum<T: TensorElem, D: Device + 'static>(
        &self,
        tensor: &Tensor<T, 2, D>,
    ) -> Result<Tensor<T, 2, D>>;

    /// Performs an All-Gather on the given tensor along a dimension.
    ///
    /// # Arguments
    /// * `tensor` - The input tensor.
    /// * `dim` - The dimension to gather along.
    fn all_gather<T: TensorElem, D: Device + 'static>(
        &self,
        tensor: &Tensor<T, 2, D>,
        dim: usize,
    ) -> Result<Tensor<T, 2, D>>;
}
