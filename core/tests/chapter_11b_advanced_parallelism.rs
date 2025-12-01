use std::sync::{Arc, Mutex};
use xla_rs::distributed::backend::CollectiveBackend;
use xla_rs::distributed::cpu_backend::CpuBackend;
use xla_rs::distributed::fsdp::{Fsdp, FsdpStrategy, Shardable};
use xla_rs::distributed::pipeline::{Pipeline, PipelineStage};
use xla_rs::tensor::{Cpu, Device, Result, Tensor, TensorElem};

// --- Helper: Simple Linear Layer for Testing ---
struct SimpleLinear {
    weight: Tensor<f32, 2, Cpu>,
    bias: Tensor<f32, 1, Cpu>,
}

impl SimpleLinear {
    fn new() -> Self {
        Self {
            weight: Tensor::ones([2, 2]),
            bias: Tensor::zeros([2]),
        }
    }
}

impl Shardable for SimpleLinear {
    fn shard<B: CollectiveBackend>(&mut self, _backend: &B) -> Result<()> {
        // In a real implementation, we would split the weight tensor here.
        // For this test, we'll just modify it to prove shard was called.
        // Let's set the first element to 0.5 to indicate "sharded" state.
        self.weight.data_mut()[0] = 0.5;
        Ok(())
    }

    fn gather<B: CollectiveBackend>(&mut self, _backend: &B) -> Result<()> {
        // Restore full weight
        self.weight.data_mut()[0] = 1.0;
        Ok(())
    }

    fn discard(&mut self) -> Result<()> {
        // Return to sharded state
        self.weight.data_mut()[0] = 0.5;
        Ok(())
    }
}

// --- Helper: Pipeline Stage ---
struct LinearStage {
    layer: SimpleLinear,
}

impl LinearStage {
    fn new() -> Self {
        Self {
            layer: SimpleLinear::new(),
        }
    }
}

impl PipelineStage<f32, Cpu> for LinearStage {
    fn forward(&self, input: &Tensor<f32, 2, Cpu>) -> Result<Tensor<f32, 2, Cpu>> {
        // Simple matmul: input @ weight.T + bias
        // For simplicity in this test, just return input + 1.0 to verify flow
        let ones = Tensor::<f32, 2>::ones(*input.shape());
        Ok((input + &ones)?)
    }
}

#[test]
fn test_fsdp_integration() -> Result<()> {
    // Setup CPU Backend (Rank 0 of 1)
    let (tx, rx) = crossbeam::channel::unbounded();
    let backend = CpuBackend::new(0, 1, rx, tx);

    // 1. Test Manual Strategy
    let model = SimpleLinear::new();
    let mut fsdp = Fsdp::new(model, FsdpStrategy::Manual, backend)?;

    // Verify initial sharding
    assert_eq!(fsdp.module().weight.data()[0], 0.5);

    // Run Forward
    let input = Tensor::<f32, 2>::ones([1, 2]);
    let _output = fsdp.forward(&input, |model, _input| {
        // Inside forward, weights should be gathered (1.0)
        assert_eq!(model.weight.data()[0], 1.0);
        Ok(Tensor::zeros([1, 2]))
    })?;

    // Verify discard (back to 0.5)
    assert_eq!(fsdp.module().weight.data()[0], 0.5);

    Ok(())
}

#[test]
fn test_pipeline_integration() -> Result<()> {
    // Create a 2-stage pipeline
    let stages: Vec<Box<dyn PipelineStage<f32, Cpu>>> =
        vec![Box::new(LinearStage::new()), Box::new(LinearStage::new())];
    let pipeline = Pipeline::new(stages);

    let input = Tensor::<f32, 2>::zeros([2, 2]);

    // 1. Standard Forward (micro_batches = 1)
    let output = pipeline.forward(&input, 1)?;
    // Input (0) + Stage1 (1) + Stage2 (1) = 2.0
    assert_eq!(output.data()[0], 2.0);

    // 2. Micro-batch Forward (micro_batches = 2)
    // Currently our implementation just loops, but this verifies the API
    let output_mb = pipeline.forward(&input, 2)?;
    assert_eq!(output_mb.data()[0], 2.0);

    Ok(())
}
