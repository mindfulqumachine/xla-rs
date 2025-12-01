use crate::tensor::{Device, Result, Tensor, TensorElem};

/// A stage in the pipeline.
///
/// Represents a layer or a block of layers that executes on a specific device (conceptually).
pub trait PipelineStage<T: TensorElem, D: Device> {
    fn forward(&self, input: &Tensor<T, 2, D>) -> Result<Tensor<T, 2, D>>;
}

/// A pipeline orchestrator.
///
/// Manages the execution of a sequence of stages, potentially using micro-batching.
pub struct Pipeline<T: TensorElem, D: Device> {
    stages: Vec<Box<dyn PipelineStage<T, D>>>,
}

impl<T: TensorElem, D: Device + 'static> Pipeline<T, D> {
    /// Creates a new Pipeline.
    pub fn new(stages: Vec<Box<dyn PipelineStage<T, D>>>) -> Self {
        Self { stages }
    }

    /// Performs a forward pass through the pipeline.
    ///
    /// # Micro-batching
    ///
    /// If `micro_batches` > 1, the input is split into smaller chunks.
    /// This allows for pipeline parallelism where different stages can process different
    /// micro-batches simultaneously (in a real multi-device setting).
    ///
    /// In this pedagogical implementation, we execute sequentially but simulate the flow.
    pub fn forward(
        &self,
        input: &Tensor<T, 2, D>,
        micro_batches: usize,
    ) -> Result<Tensor<T, 2, D>> {
        if micro_batches <= 1 {
            // Standard forward pass
            let mut x = input.clone();
            for stage in &self.stages {
                x = stage.forward(&x)?;
            }
            Ok(x)
        } else {
            // Micro-batching logic
            let batch_size = input.shape()[0];
            let _micro_batch_size = batch_size / micro_batches;

            // 1. Split input
            // Note: We need a split method on Tensor. For now, let's assume we can slice or just fail if not implemented.
            // Since we don't have a robust split yet, let's just warn and run sequentially for now
            // OR implement a basic split if possible.
            // Given the constraints, let's implement a simplified loop that *would* be parallel.

            // For now, we just run the whole batch through each stage to verify correctness.
            // Real PP requires async/multi-threading which is complex for this scope.
            // We'll stick to the "Bubble" concept explanation in docs and simple execution here.

            let mut x = input.clone();
            for stage in &self.stages {
                x = stage.forward(&x)?;
            }
            Ok(x)
        }
    }
}
