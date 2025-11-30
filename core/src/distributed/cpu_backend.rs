use super::backend::CollectiveBackend;
use crate::tensor::{Cpu, Result, Tensor, TensorElem};
use crossbeam::channel::{Receiver, Sender};

/// A CPU-based collective backend for educational purposes.
///
/// Implements the **Ring All-Reduce** algorithm using `crossbeam` channels.
/// This simulates the data movement that happens on GPUs via NCCL.
pub struct CpuBackend {
    rank: usize,
    world_size: usize,
    left_rx: Receiver<Vec<u8>>, // Receive from rank - 1
    right_tx: Sender<Vec<u8>>,  // Send to rank + 1
}

impl CpuBackend {
    pub fn new(
        rank: usize,
        world_size: usize,
        left_rx: Receiver<Vec<u8>>,
        right_tx: Sender<Vec<u8>>,
    ) -> Self {
        Self {
            rank,
            world_size,
            left_rx,
            right_tx,
        }
    }
}

impl CollectiveBackend for CpuBackend {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn all_reduce_sum<T: TensorElem, D: crate::tensor::Device + 'static>(
        &self,
        tensor: &Tensor<T, 2, D>,
    ) -> Result<Tensor<T, 2, D>> {
        // NOTE: In a real implementation, we would enforce D == Cpu.
        // For now, we assume the tensor is accessible on CPU.
        // We also cheat slightly by accessing .data() directly, assuming contiguous memory.

        let mut data = tensor.data().to_vec();
        let total_elements = data.len();
        let chunk_size = total_elements.div_ceil(self.world_size);

        // --- Phase 1: Scatter-Reduce ---
        // In each step, we send a chunk to the right and receive a chunk from the left.
        // We add the received chunk to our local buffer.

        for step in 0..self.world_size - 1 {
            // Calculate which chunk we are sending.
            // In Ring All-Reduce, rank r sends chunk (r - step) % world_size
            let send_chunk_idx =
                (self.rank as isize - step as isize).rem_euclid(self.world_size as isize) as usize;
            let recv_chunk_idx = (self.rank as isize - step as isize - 1)
                .rem_euclid(self.world_size as isize) as usize;

            let start = send_chunk_idx * chunk_size;
            let end = (start + chunk_size).min(total_elements);
            let send_data = &data[start..end];

            // Send to right
            // We cast to u8 for generic transmission (unsafe but educational)
            let send_bytes = unsafe {
                std::slice::from_raw_parts(
                    send_data.as_ptr() as *const u8,
                    std::mem::size_of_val(send_data),
                )
            };
            self.right_tx.send(send_bytes.to_vec()).unwrap();

            // Receive from left
            let recv_bytes = self.left_rx.recv().unwrap();
            let recv_data = unsafe {
                std::slice::from_raw_parts(
                    recv_bytes.as_ptr() as *const T,
                    recv_bytes.len() / std::mem::size_of::<T>(),
                )
            };

            // Reduce (Add)
            let recv_start = recv_chunk_idx * chunk_size;
            for (i, &val) in recv_data.iter().enumerate() {
                if recv_start + i < total_elements {
                    data[recv_start + i] += val;
                }
            }
        }

        // --- Phase 2: All-Gather ---
        // Now each rank has one fully reduced chunk. We need to share it with everyone.
        for step in 0..self.world_size - 1 {
            let send_chunk_idx = (self.rank as isize - step as isize + 1)
                .rem_euclid(self.world_size as isize) as usize;
            let recv_chunk_idx =
                (self.rank as isize - step as isize).rem_euclid(self.world_size as isize) as usize;

            let start = send_chunk_idx * chunk_size;
            let end = (start + chunk_size).min(total_elements);
            let send_data = &data[start..end];

            let send_bytes = unsafe {
                std::slice::from_raw_parts(
                    send_data.as_ptr() as *const u8,
                    std::mem::size_of_val(send_data),
                )
            };
            self.right_tx.send(send_bytes.to_vec()).unwrap();

            let recv_bytes = self.left_rx.recv().unwrap();
            let recv_data = unsafe {
                std::slice::from_raw_parts(
                    recv_bytes.as_ptr() as *const T,
                    recv_bytes.len() / std::mem::size_of::<T>(),
                )
            };

            let recv_start = recv_chunk_idx * chunk_size;
            for (i, &val) in recv_data.iter().enumerate() {
                if recv_start + i < total_elements {
                    data[recv_start + i] = val;
                }
            }
        }

        // Reconstruct tensor
        // Note: This assumes D is Cpu. If D is XlaDevice, we'd need to transfer back.
        // Since this is CpuBackend, we just return a new Cpu tensor.
        // Ideally we would use Tensor::from_vec but we need to respect D.
        // For this educational snippet, we panic if D is not Cpu.
        if std::any::TypeId::of::<D>() != std::any::TypeId::of::<Cpu>() {
            panic!("CpuBackend only supports Cpu tensors");
        }

        // Hack to return generic D tensor (we know it's Cpu)
        // In real code we'd use a trait method on Device to create tensor from data.
        // Here we just return Ok(unsafe { std::mem::transmute(Tensor::new(data, *tensor.shape()).unwrap()) })
        // But that's too unsafe. Let's just return a Result and assume the caller knows.

        // Safe fallback:
        // We verified D is Cpu above (via TypeId check, though we panicked if not).
        // We need to return Tensor<T, 2, D>.
        // Since we know D is Cpu, Tensor<T, 2, D> is layout-compatible with Tensor<T, 2, Cpu>.
        let result_cpu = Tensor::<T, 2, Cpu>::new(data, *tensor.shape()).unwrap();

        // Use pointer casting to bypass the type system's size check for generic D
        let result_ptr = &result_cpu as *const Tensor<T, 2, Cpu> as *const Tensor<T, 2, D>;
        let result = unsafe { result_ptr.read() };
        std::mem::forget(result_cpu); // Prevent double-free since we moved ownership via read()

        Ok(result)
    }

    fn all_gather<T: TensorElem, D: crate::tensor::Device + 'static>(
        &self,
        _tensor: &Tensor<T, 2, D>,
        _dim: usize,
    ) -> Result<Tensor<T, 2, D>> {
        unimplemented!("All-Gather not yet implemented for CpuBackend")
    }
}
