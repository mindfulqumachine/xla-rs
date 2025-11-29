# Chapter 2: Tensors (The Bedrock)

## Part 1: The Concept

At the heart of every deep learning framework lies the **Tensor**. A tensor is simply a container for numbers, organized into a grid. It is a generalization of vectors and matrices to *`N`* dimensions.

### Why Tensors?

Deep Learning involves performing massive amounts of mathematical operations (addition, multiplication) on huge datasets. Tensors provide a uniform way to represent this data so that we can process it efficiently, often in parallel on hardware like GPUs.

### Modeling the Real World

Everything in the physical world can be represented as a tensor. The number of dimensions (or **Rank**) depends on the complexity of the data.

#### Rank 0: Scalar
A single number.
-   **Example**: The current temperature (23.5°C), the loss value of a model (0.001).
-   **Shape**: `[]`

#### Rank 1: Vector
A list of numbers.
-   **Example**: A time series of a stock price over 10 days.
-   **Shape**: `[10]`
-   **Visual**: `[120.5, 121.0, 119.8, ...]`

#### Rank 2: Matrix
A 2D grid of numbers (Rows and Columns).
-   **Example**: A grayscale image (Height x Width). Each pixel is a number representing brightness.
-   **Shape**: `[28, 28]` (for a standard MNIST digit).

#### Rank 3: Volume
A 3D cube of numbers.
-   **Example**: A color image. It has Height, Width, and **3 Color Channels** (Red, Green, Blue).
-   **Shape**: `[256, 256, 3]`

#### Rank 4: Batch of Volumes
A collection of 3D objects.
-   **Example**: A batch of 32 color images processed together for training efficiency.
-   **Shape**: `[32, 256, 256, 3]` (Batch Size, Height, Width, Channels).

---

## Part 2: The Implementation

In `xla-rs`, we don't just want to store these numbers; we want to do it **safely** and **fast**. We leverage Rust's powerful type system to catch errors at compile time that would normally crash your program at runtime in other languages.

### The `Tensor` Struct

Our `Tensor` struct is defined in [`src/tensor/mod.rs`](https://github.com/mindfulqumachine/xla-rs/blob/main/core/src/tensor/mod.rs). Notice how the definition mirrors the conceptual structure:

```rust
# extern crate xla_rs;
# use xla_rs::tensor::{Device, Cpu, TensorElem};
/// The core Tensor struct.
///
/// # Generics
///
/// - `T`: The element type (must implement `TensorElem`).
/// - `RANK`: The number of dimensions (const generic).
/// - `D`: The device where data is stored (defaults to `Cpu`).
pub struct Tensor<T, const RANK: usize, D: Device = Cpu>
where
    T: TensorElem,
{
    shape: [usize; RANK],
    strides: [usize; RANK],
    data: D::Storage<T>,
    device: D,
}
```

### Power of the Type System

1.  **Const Generics (`const RANK: usize`)**: We encode the dimensionality of the tensor directly into its type. This means a `Tensor<f32, 2>` (matrix) is a distinct type from `Tensor<f32, 3>` (3D volume). This allows the compiler to catch rank-mismatch errors before your code even runs.
2.  **`TensorElem` Trait**: The `T: TensorElem` bound ensures that we only store valid numerical types (like `f32`, `f64`, `i32`) that support arithmetic operations and are safe to send across threads (`Send + Sync`).
3.  **`Device` Trait**: The `D: Device` generic allows us to abstract over where the data lives (CPU, GPU, TPU) without changing the core logic.

### Understanding Dimensions and Shape

A common source of confusion is: *Which dimension corresponds to what?* Why is `shape[0]` usually the rows and `shape[1]` the columns?

In `xla-rs` (and most modern frameworks like PyTorch and NumPy), this is determined by the **Row-Major** convention. This isn't a physical law, but a standard way of organizing data.

**The "Outermost to Innermost" Rule:**
We list dimensions from the **most significant** (slowest changing, outermost container) to the **least significant** (fastest changing, contiguous elements).

-   **Rank 2 (Matrix)**: Think of a matrix as a "list of rows".
    -   **Dim 0 (Outer)**: Which row are we in?
    -   **Dim 1 (Inner)**: Which column in that row?
    -   Shape: `[Rows, Cols]`

-   **Rank 3 (Volume)**: Think of a book (a list of pages, where each page is a matrix).
    -   **Dim 0**: Which page? (Depth)
    -   **Dim 1**: Which row on the page? (Height)
    -   **Dim 2**: Which character in the row? (Width)
    -   Shape: `[Depth, Height, Width]`

-   **Rank N**: `[d_0, d_1, ..., d_{N-1}]`
    -   `d_0` is the coarsest container (e.g., Batch Size).
    -   `d_{N-1}` is the finest detail (e.g., Color Channels), stored contiguously in memory.

### Memory Layout: Shapes and Strides

Computer memory is fundamentally linear (one-dimensional). It's just a long strip of bytes. However, Tensors are N-dimensional.

**Strides** are the bridge that allows us to map an N-dimensional logical index `(i, j, k)` to a 1-dimensional physical index in memory.

The **stride** for a specific dimension tells us: *"How many elements do I need to skip in the flat memory array to move one step along this dimension?"*

#### Example: A `[2, 3]` Matrix

Consider a matrix with 2 rows and 3 columns:

$$
\begin{bmatrix}
A & B & C \\\\
D & E & F
\end{bmatrix}
$$

In memory, this is stored as a flat vector: `[A, B, C, D, E, F]`.

-   **Moving along columns (dimension 1)**: To go from *A* to *B*, we move **1** step in memory. So, `strides[1] = 1`.
-   **Moving along rows (dimension 0)**: To go from *A* to *D*, we need to skip the entire first row (*A*, *B*, *C*). That's **3** elements. So, `strides[0] = 3`.

The formula to calculate the flat index for logical index `(i, j)` is:
$$ \texttt{flat\_index} = i \times \texttt{strides}\[0\] + j \times \texttt{strides}\[1\] $$

For *E* at `(1, 1)`:
$$ 1 \times 3 + 1 \times 1 = 4 $$
Checking our flat vector `[A, B, C, D, E, F]`, index 4 is indeed *E*.

#### Generalizing to N Dimensions

The concept extends naturally to any number of dimensions. For a tensor of shape `[d_0, d_1, ..., d_{N-1}]`, the stride for dimension `k` (`strides[k]`) is the product of the sizes of all subsequent dimensions:

$$ \texttt{strides}\[k\] = \prod_{m=k+1}^{N-1} \texttt{shape}\[m\] $$

*(Note: The last stride `strides[N-1]` is always 1 for standard contiguous memory).*

**Example: A 3D Tensor `[2, 2, 3]`**
Shape: `[Depth=2, Rows=2, Cols=3]`
Total elements: \\(2 \times 2 \times 3 = 12\\).

-   **Dimension 2 (Cols)**: Moving 1 step changes the column. `strides[2] = 1`.
-   **Dimension 1 (Rows)**: Moving 1 step skips a whole row of columns. `strides[1] = shape[2] = 3`.
-   **Dimension 0 (Depth)**: Moving 1 step skips a whole 2D slice (matrix). `strides[0] = shape[1] \times shape[2] = 2 \times 3 = 6`.

The flat index for logical index `(i, j, k)` is:
$$ \texttt{flat\_index} = i \times \texttt{strides}\[0\] + j \times \texttt{strides}\[1\] + k \times \texttt{strides}\[2\] $$
$$ \texttt{flat\_index} = i \times 6 + j \times 3 + k \times 1 $$

### Hands On

Let's create some tensors and see the type system in action.

```rust
# extern crate xla_rs;
use xla_rs::tensor::Tensor;

fn main() {
    // 1. Create a 2D Tensor (Matrix) from a vector
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let t = Tensor::<f32, 2>::new(data, [2, 2]).unwrap();
    println!("Matrix:\n{:?}", t);

    // 2. Create a Tensor of zeros
    let zeros = Tensor::<f32, 3>::zeros([2, 2, 2]);
    println!("Zeros:\n{:?}", zeros);

    // 3. Reshape
    // Note: The total number of elements must remain the same.
    // The new rank is inferred or specified in the type.
    let reshaped: Tensor<f32, 1> = t.reshape([4]).unwrap();
    println!("Reshaped to vector:\n{:?}", reshaped);
}
```

### Compile-Time Tensors

One of the unique features of `xla-rs` is the ability to create and manipulate tensors entirely at compile time. This is powered by the `ConstDevice`.

#### Zero-Overhead Creation

You can define tensors as `const` variables. The data is embedded directly into the binary, requiring no runtime allocation or initialization.

```rust
# extern crate xla_rs;
use xla_rs::tensor::{Tensor, ConstDevice};

fn main() {
    // A 2x2 matrix created at compile time
    const A: Tensor<f32, 2, ConstDevice<4>> = Tensor::new_const(
        [1.0, 2.0, 3.0, 4.0], 
        [2, 2]
    );
    
    println!("Const Tensor: {:?}", A);
}
```

#### Compile-Time Operations

Operations like `transpose` and `matmul` can also be performed at compile time. This means the complex calculations happen during the build process, and the runtime program simply loads the pre-computed result from memory.

```rust
# extern crate xla_rs;
use xla_rs::tensor::{Tensor, ConstDevice};

fn main() {
    const A: Tensor<f32, 2, ConstDevice<4>> = Tensor::new_const([1.0, 2.0, 3.0, 4.0], [2, 2]);
    
    // Transpose is computed by the compiler!
    // Runtime cost: 0 (just memory access)
    const B: Tensor<f32, 2, ConstDevice<4>> = A.transpose();
    
    println!("Transposed: {:?}", B);
}
```

This is particularly useful for inference-only models where weights are fixed. We can pre-compute transformations (like transposing weight matrices) so the device doesn't have to do it at runtime.

#### Compile-Time Safety

Because of `const RANK`, if you try to treat a 2D tensor as 3D without explicit reshaping, the compiler will stop you.

```rust,compile_fail
# extern crate xla_rs;
# use xla_rs::tensor::Tensor;
fn takes_3d_tensor(t: Tensor<f32, 3>) {}

let t = Tensor::<f32, 2>::zeros([2, 2]);
// This won't compile! Expected Tensor<_, 3>, found Tensor<_, 2>
takes_3d_tensor(t); 
```

### Matrix Multiplication and Broadcasting

In `xla-rs`, matrix multiplication (`matmul`) is a powerful operation that supports more than just 2D matrices. It implements **Batched Matrix Multiplication** for tensors with Rank > 2.

#### How it works

For any tensor of Rank \\(N \ge 2\\), the first \\(N-2\\) dimensions are treated as **batch dimensions**, and the multiplication is performed on the last two dimensions.

-   **Input A**: `[Batch..., M, K]`
-   **Input B**: `[Batch..., K, N]`
-   **Output**: `[Batch..., M, N]`

This corresponds to the Einstein summation: `...mk,...kn->...mn`.

#### Example: Multi-Head Attention

In Transformers, we often work with 4D tensors representing `[Batch, Heads, Sequence, Hidden]`.

```rust
# extern crate xla_rs;
# use xla_rs::tensor::Tensor;
fn main() {
    // Shape: [Batch=1, Heads=2, M=2, K=3]
    let a = Tensor::<f32, 4>::ones([1, 2, 2, 3]);
    
    // Shape: [Batch=1, Heads=2, K=3, N=2]
    let b = Tensor::<f32, 4>::ones([1, 2, 3, 2]);
    
    // Result: [Batch=1, Heads=2, M=2, N=2]
    let c = a.matmul(&b).unwrap();
    
    println!("Output shape: {:?}", c.shape());
}
```
