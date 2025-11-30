use crossbeam::channel::unbounded;
use std::thread;
use xla_rs::distributed::backend::CollectiveBackend;
use xla_rs::distributed::cpu_backend::CpuBackend;
use xla_rs::tensor::{Cpu, Tensor};

#[test]
fn test_ring_all_reduce_cpu() {
    let world_size = 4;
    let mut handles = vec![];

    // Create channels for the ring: 0->1->2->3->0
    let mut txs = vec![];
    let mut rxs = vec![];

    for _ in 0..world_size {
        let (tx, rx) = unbounded();
        txs.push(tx);
        rxs.push(rx);
    }

    for rank in 0..world_size {
        // Left neighbor is rank - 1 (or world_size - 1)
        // Right neighbor is rank + 1 (or 0)
        // My receiver is rxs[rank] (which receives from left)
        // My sender is txs[(rank + 1) % world_size] (which sends to right)

        // Wait, the CpuBackend expects:
        // left_rx: Receive from rank - 1
        // right_tx: Send to rank + 1

        // So I need to give rank `i`:
        // left_rx = rxs[i] (where txs[i-1] sends to)
        // right_tx = txs[i+1] (where I send to)

        // Let's rewire:
        // Channel i connects Node i to Node i+1.
        // Node i sends on Channel i.
        // Node i+1 receives on Channel i.

        let my_tx_idx = rank;
        let my_rx_idx = (rank + world_size - 1) % world_size;

        let right_tx = txs[my_tx_idx].clone();
        let left_rx = rxs[my_rx_idx].clone();

        let handle = thread::spawn(move || {
            let backend = CpuBackend::new(rank, world_size, left_rx, right_tx);

            // Create a tensor: [1.0, 1.0, ...] * (rank + 1)
            // So Rank 0 has 1s, Rank 1 has 2s, etc.
            // Sum should be 1+2+3+4 = 10.
            let size = 8;
            let data = vec![(rank + 1) as f32; size];
            let tensor = Tensor::<f32, 2, Cpu>::new(data, [2, 4]).unwrap();

            let result = backend.all_reduce_sum(&tensor).unwrap();

            // Verify
            let expected = 10.0;
            for &val in result.data() {
                assert!(
                    (val - expected).abs() < 1e-5,
                    "Rank {}: Expected {}, got {}",
                    rank,
                    expected,
                    val
                );
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_cpu_backend_properties() {
    let (tx, rx) = unbounded();
    let backend = CpuBackend::new(2, 4, rx, tx);
    assert_eq!(backend.rank(), 2);
    assert_eq!(backend.world_size(), 4);
}

#[test]
#[should_panic(expected = "All-Gather not yet implemented for CpuBackend")]
fn test_cpu_backend_all_gather_panic() {
    let (tx, rx) = unbounded();
    let backend = CpuBackend::new(0, 1, rx, tx);
    let data = vec![1.0f32];
    let tensor = Tensor::<f32, 2, Cpu>::new(data, [1, 1]).unwrap();
    let _ = backend.all_gather(&tensor, 0);
}

#[test]
#[should_panic(expected = "CpuBackend only supports Cpu tensors")]
fn test_cpu_backend_non_cpu_device_panic() {
    let (tx, rx) = unbounded();
    let backend = CpuBackend::new(0, 1, rx, tx);

    // Create a tensor on ConstDevice
    let data = [1.0f32];
    let tensor = Tensor::<f32, 2, xla_rs::tensor::ConstDevice<1>>::new_const(data, [1, 1]);
    let _ = backend.all_reduce_sum(&tensor);
}
