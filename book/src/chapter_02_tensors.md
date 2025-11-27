# Chapter 2: Tensors (The Bedrock)

## Part 1: The Concept

At the heart of every deep learning framework lies the **Tensor**. While the name might sound intimidating, a tensor is simply a container for numbers, organized into a grid. It is a generalization of vectors and matrices to *N* dimensions.

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

Our `Tensor` struct is defined in [`src/tensor/mod.rs`](../../src/tensor/mod.rs). Notice how the definition mirrors the conceptual structure:

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

### Memory Layout: Shapes and Strides

Computer memory is fundamentally linear (one-dimensional). It's just a long strip of bytes. However, Tensors are N-dimensional.

**Strides** are the bridge that allows us to map an N-dimensional logical index `(i, j, k)` to a 1-dimensional physical index in memory.

The **stride** for a specific dimension tells us: *"How many elements do I need to skip in the flat memory array to move one step along this dimension?"*

#### Example: A [2, 3] Matrix

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
$$ \texttt{flat\_index} = i \times \texttt{strides}[0] + j \times \texttt{strides}[1] $$

For *E* at `(1, 1)`:
$$ 1 \times 3 + 1 \times 1 = 4 $$
Checking our flat vector `[A, B, C, D, E, F]`, index 4 is indeed *E*.

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

#### Compile-Time Safety

Because of `const RANK`, if you try to treat a 2D tensor as 3D without explicit reshaping, the compiler will stop you.

```rust,compile_fail
# extern crate xla_rs;
# use xla_rs::tensor::Tensor;
fn takes_3d_tensor(t: Tensor<f32, 3>) {}

let t = Tensor::<f32, 2>::zeros([2, 2]);
// This won't compile! Expected Tensor<_, 3>, found Tensor<_, 2>
// takes_3d_tensor(t); 
```
