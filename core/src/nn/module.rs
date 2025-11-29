use crate::tensor::TensorElem;
use std::fmt::Debug;

/// A Module trait for Neural Network layers.
///
/// # Why is this needed?
///
/// The `Module` trait serves as the fundamental building block for all neural network components
/// in `xla-rs`. It enforces a common interface that ensures:
///
/// 1.  **Thread Safety**: By requiring `Send` and `Sync`, we ensure that models can be safely
///     shared across threads (e.g., for parallel inference or data loading).
/// 2.  **Debuggability**: Requiring `Debug` ensures that the structure of any model can be
///     easily inspected.
/// 3.  **Extensibility**: While currently a marker trait, this abstraction allows us to generically
///     implement features like parameter counting, device movement, and serialization for all layers
///     in the future without breaking changes.
pub trait Module<T: TensorElem>: Debug + Send + Sync {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct MockModule;

    impl Module<f32> for MockModule {}

    #[test]
    fn test_module_implementation() {
        let module = MockModule;
        // Just verify it compiles and runs
        println!("{:?}", module);
    }
}
