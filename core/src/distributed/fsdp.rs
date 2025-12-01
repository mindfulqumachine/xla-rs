use crate::distributed::backend::CollectiveBackend;
use crate::tensor::{Device, Result, Tensor, TensorElem};

/// Strategies for Fully Sharded Data Parallelism.
#[derive(Debug, Clone, Copy)]
pub enum FsdpStrategy {
    /// Pedagogical implementation: Manually shards, gathers, and discards parameters.
    ///
    /// This mode is slow but educational. It demonstrates the FSDP lifecycle:
    /// 1. `gather`: All-Gather weights from all ranks.
    /// 2. `forward`: Compute with full weights.
    /// 3. `discard`: Drop full weights, keeping only the local shard.
    Manual,

    /// Production implementation: Uses XLA's GSPMD (General Spmd Partitioning).
    ///
    /// In this mode, we annotate tensors with sharding specs, and the XLA compiler
    /// automatically inserts the necessary collective operations.
    Xla,
}

/// A trait for modules that can be sharded for FSDP.
///
/// Since Rust traits don't allow iterating over fields easily without macros,
/// we require modules to implement this trait to participate in `Manual` FSDP.
pub trait Shardable {
    /// Shards the parameters of the module.
    ///
    /// This should be called once at initialization.
    fn shard<B: CollectiveBackend>(&mut self, backend: &B) -> Result<()>;

    /// Gathers the full parameters from all ranks.
    ///
    /// This is called before the forward pass.
    fn gather<B: CollectiveBackend>(&mut self, backend: &B) -> Result<()>;

    /// Discards the full parameters, keeping only the local shard.
    ///
    /// This is called after the forward pass to free memory.
    fn discard(&mut self) -> Result<()>;
}

/// A wrapper for Fully Sharded Data Parallelism.
///
/// # Example
///
/// ```rust,ignore
/// let backend = CpuBackend::new(...);
/// let model = MyModel::new();
/// let mut fsdp_model = Fsdp::new(model, FsdpStrategy::Manual, backend);
///
/// // Forward pass
/// let output = fsdp_model.forward(&input)?;
/// ```
pub struct Fsdp<M, B> {
    module: M,
    strategy: FsdpStrategy,
    backend: B,
}

impl<M, B> Fsdp<M, B>
where
    M: Shardable,
    B: CollectiveBackend,
{
    /// Creates a new FSDP wrapper.
    pub fn new(mut module: M, strategy: FsdpStrategy, backend: B) -> Result<Self> {
        // Initial sharding
        if let FsdpStrategy::Manual = strategy {
            module.shard(&backend)?;
        }
        // For Xla strategy, we assume the user handles sharding annotations externally
        // or we would add a `shard_xla` method here.

        Ok(Self {
            module,
            strategy,
            backend,
        })
    }

    /// Performs the forward pass with FSDP logic.
    pub fn forward<T: TensorElem, D: Device, F>(
        &mut self,
        input: &Tensor<T, 2, D>,
        forward_fn: F,
    ) -> Result<Tensor<T, 2, D>>
    where
        F: FnOnce(&M, &Tensor<T, 2, D>) -> Result<Tensor<T, 2, D>>,
    {
        match self.strategy {
            FsdpStrategy::Manual => {
                // 1. All-Gather
                self.module.gather(&self.backend)?;

                // 2. Compute
                let result = forward_fn(&self.module, input);

                // 3. Discard
                // We discard even if forward failed, to avoid memory leaks?
                // Actually if forward fails we might panic or return err.
                // Let's try to discard.
                self.module.discard()?;

                result
            }
            FsdpStrategy::Xla => {
                // In XLA mode, the compiler handles everything.
                // We just run the forward pass.
                forward_fn(&self.module, input)
            }
        }
    }

    /// Access the inner module.
    pub fn module(&self) -> &M {
        &self.module
    }

    /// Access the inner module mutably.
    pub fn module_mut(&mut self) -> &mut M {
        &mut self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Cpu, Tensor};
    use std::sync::{Arc, Mutex};

    // Mock CollectiveBackend to verify calls
    struct MockBackend {
        rank: usize,
        world_size: usize,
        log: Arc<Mutex<Vec<String>>>,
    }

    impl MockBackend {
        fn new(rank: usize, log: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                rank,
                world_size: 2,
                log,
            }
        }
    }

    impl CollectiveBackend for MockBackend {
        fn rank(&self) -> usize {
            self.rank
        }

        fn world_size(&self) -> usize {
            self.world_size
        }

        fn all_reduce_sum<T: TensorElem, D: Device + 'static>(
            &self,
            tensor: &Tensor<T, 2, D>,
        ) -> Result<Tensor<T, 2, D>> {
            self.log.lock().unwrap().push("all_reduce_sum".to_string());
            Ok(tensor.clone())
        }

        fn all_gather<T: TensorElem, D: Device + 'static>(
            &self,
            tensor: &Tensor<T, 2, D>,
            _dim: usize,
        ) -> Result<Tensor<T, 2, D>> {
            self.log.lock().unwrap().push("all_gather".to_string());
            Ok(tensor.clone())
        }
    }

    // Mock Shardable Module
    struct MockModule {
        _param: Tensor<f32, 2, Cpu>,
        log: Arc<Mutex<Vec<String>>>,
    }

    impl MockModule {
        fn new(log: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                _param: Tensor::zeros([2, 2]),
                log,
            }
        }
    }

    impl Shardable for MockModule {
        fn shard<B: CollectiveBackend>(&mut self, _backend: &B) -> Result<()> {
            self.log.lock().unwrap().push("shard".to_string());
            Ok(())
        }

        fn gather<B: CollectiveBackend>(&mut self, _backend: &B) -> Result<()> {
            self.log.lock().unwrap().push("gather".to_string());
            Ok(())
        }

        fn discard(&mut self) -> Result<()> {
            self.log.lock().unwrap().push("discard".to_string());
            Ok(())
        }
    }

    #[test]
    fn test_fsdp_manual_lifecycle() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let backend = MockBackend::new(0, log.clone());
        let module = MockModule::new(log.clone());

        // 1. Creation (should trigger shard)
        let mut fsdp = Fsdp::new(module, FsdpStrategy::Manual, backend).unwrap();
        assert_eq!(*log.lock().unwrap(), vec!["shard"]);
        log.lock().unwrap().clear();

        // 2. Forward (should trigger gather -> forward -> discard)
        let input = Tensor::<f32, 2>::zeros([1, 2]);
        let _ = fsdp
            .forward(&input, |_, input| {
                log.lock().unwrap().push("forward_fn".to_string());
                Ok(input.clone())
            })
            .unwrap();

        assert_eq!(
            *log.lock().unwrap(),
            vec!["gather", "forward_fn", "discard"]
        );
    }

    #[test]
    fn test_fsdp_xla_lifecycle() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let backend = MockBackend::new(0, log.clone());
        let module = MockModule::new(log.clone());

        // 1. Creation (should NOT trigger shard in Xla mode)
        let mut fsdp = Fsdp::new(module, FsdpStrategy::Xla, backend).unwrap();
        assert!(log.lock().unwrap().is_empty());

        // 2. Forward (should ONLY trigger forward_fn)
        let input = Tensor::<f32, 2>::zeros([1, 2]);
        let _ = fsdp
            .forward(&input, |_, input| {
                log.lock().unwrap().push("forward_fn".to_string());
                Ok(input.clone())
            })
            .unwrap();

        assert_eq!(*log.lock().unwrap(), vec!["forward_fn"]);
    }

    #[test]
    fn test_accessors() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let backend = MockBackend::new(0, log.clone());
        let module = MockModule::new(log.clone());
        let mut fsdp = Fsdp::new(module, FsdpStrategy::Manual, backend).unwrap();

        // Test module()
        let _ = fsdp.module();

        // Test module_mut()
        let _ = fsdp.module_mut();
    }
}
