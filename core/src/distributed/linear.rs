use super::comm::all_reduce_sum;
use xla::{Result, XlaOp};

/// Defines how a linear layer is split across devices.
///
/// In Tensor Parallelism, we split the weight matrix $W$ to fit it into memory.
///
/// # 🧠 The Math
///
/// A linear layer performs $Y = X W$.
/// If we split $W$ into two parts $W_1$ and $W_2$:
///
/// ## Column Parallelism
/// We split $W$ along columns: $W = [W_1 | W_2]$.
/// $$
/// Y = X [W_1 | W_2] = [X W_1 | X W_2] = [Y_1 | Y_2]
/// $$
/// - Each rank computes a part of the *output vector*.
/// - **Communication**: None in forward pass. All-Gather needed if we want full $Y$.
///
/// ## Row Parallelism
/// We split $W$ along rows: $W = \begin{bmatrix} W_1 \\ W_2 \end{bmatrix}$.
/// We must also split $X$ along columns: $X = [X_1 | X_2]$.
/// $$
/// Y = [X_1 | X_2] \begin{bmatrix} W_1 \\ W_2 \end{bmatrix} = X_1 W_1 + X_2 W_2 = Y_1 + Y_2
/// $$
/// - Each rank computes a *partial sum* of the output.
/// - **Communication**: **All-Reduce (Sum)** is required to get the final $Y$.
#[derive(Debug, Clone, Copy)]
pub enum ParallelStrategy {
    /// Column Parallelism.
    ///
    /// Splits the weight matrix $W$ along the **output** dimension (columns).
    /// - Input $X$ is replicated (same on all ranks).
    /// - Output $Y$ is split (each rank holds a part of the output vector).
    Column,

    /// Row Parallelism.
    ///
    /// Splits the weight matrix $W$ along the **input** dimension (rows).
    /// - Input $X$ is split (each rank holds a part of the feature vector).
    /// - Output $Y$ is a partial sum.
    /// - **Requires All-Reduce** to sum the partial results.
    Row,
}

/// A Linear layer with Tensor Parallelism support.
///
/// This layer automatically handles the communication required for distributed matrix multiplication.
///
/// # Example Usage (Conceptual)
///
/// ```rust,ignore
/// // Rank 0
/// let layer = TensorParallelLinear::new(w1, None, ParallelStrategy::Column);
/// let y1 = layer.forward(&x)?; // y1 is the first half of the output
///
/// // Rank 1
/// let layer = TensorParallelLinear::new(w2, None, ParallelStrategy::Column);
/// let y2 = layer.forward(&x)?; // y2 is the second half of the output
/// ```
pub struct TensorParallelLinear {
    pub weight: XlaOp,
    pub bias: Option<XlaOp>,
    pub strategy: ParallelStrategy,
}

impl TensorParallelLinear {
    /// Creates a new Tensor Parallel Linear layer.
    ///
    /// # Arguments
    /// * `weight` - The local shard of the weight matrix.
    /// * `bias` - The local shard of the bias (if applicable).
    /// * `strategy` - The parallelism strategy (Column or Row).
    pub fn new(weight: XlaOp, bias: Option<XlaOp>, strategy: ParallelStrategy) -> Self {
        Self {
            weight,
            bias,
            strategy,
        }
    }

    /// Performs the forward pass.
    ///
    /// - **Column Parallel**: Computes $X W_i$. No communication.
    /// - **Row Parallel**: Computes $X_i W_i$ and performs **All-Reduce Sum**.
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
