use super::comm::all_reduce_sum;
use xla::{Result, XlaOp};

#[derive(Debug, Clone, Copy)]
pub enum ParallelStrategy {
    /// Column Parallelism: Splits the weight matrix along the output dimension.
    /// Input X is replicated. Output Y is split.
    /// No communication required in forward pass.
    Column,

    /// Row Parallelism: Splits the weight matrix along the input dimension.
    /// Input X must be split along the feature dimension.
    /// Output Y is a partial sum, requiring an All-Reduce to finalize.
    Row,
}

pub struct TensorParallelLinear {
    pub weight: XlaOp,
    pub bias: Option<XlaOp>,
    pub strategy: ParallelStrategy,
}

impl TensorParallelLinear {
    pub fn new(weight: XlaOp, bias: Option<XlaOp>, strategy: ParallelStrategy) -> Self {
        Self {
            weight,
            bias,
            strategy,
        }
    }

    pub fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        // Basic matrix multiplication: Y = X @ W
        // Note: XLA's dot_general or matmul might be needed depending on dimensions
        let output = x.dot(&self.weight)?;

        match self.strategy {
            ParallelStrategy::Column => {
                // Output is split. Bias should also be split.
                // If bias exists, add it.
                if let Some(bias) = &self.bias {
                    // Bias addition is local
                    Ok((output + bias)?)
                } else {
                    Ok(output)
                }
            }
            ParallelStrategy::Row => {
                // Output is a partial sum.
                // If bias exists, it is usually added ONLY on one rank or added after reduction.
                // However, in standard Megatron RowParallel, bias is added after the All-Reduce.
                // Let's assume bias is handled after reduction for correctness.

                // 1. All-Reduce Sum
                let reduced_output = all_reduce_sum(&output)?;

                // 2. Add Bias (if any)
                if let Some(bias) = &self.bias {
                    Ok((reduced_output + bias)?)
                } else {
                    Ok(reduced_output)
                }
            }
        }
    }
}
