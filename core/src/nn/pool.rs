use crate::tensor::{Result, Tensor, TensorElem};

/// 2D Max Pooling Layer.
pub struct MaxPool2d {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
}

impl MaxPool2d {
    /// Creates a new MaxPool2d layer.
    ///
    /// # Arguments
    ///
    /// * `kernel_size` - Size of the pooling window.
    /// * `stride` - Stride of the pooling.
    /// * `padding` - Zero-padding added to both sides of the input.
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size: [kernel_size, kernel_size],
            stride: [stride, stride],
            padding: [padding, padding],
        }
    }

    /// Performs the forward pass.
    pub fn forward<T: TensorElem>(
        &self,
        input: &Tensor<T, 4, crate::tensor::Cpu>,
    ) -> Result<Tensor<T, 4, crate::tensor::Cpu>> {
        input.max_pool2d(self.kernel_size, self.stride, self.padding)
    }
}
