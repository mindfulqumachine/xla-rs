use std::collections::HashMap;
use xla_rs::checkpoint::{load_checkpoint, save_checkpoint};
use xla_rs::optim::{AdamW, Optimizer};
use xla_rs::tensor::Tensor;

#[test]
fn test_save_load_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("model.safetensors");

    let t1 = Tensor::<f32, 1>::new(vec![1.0, 2.0, 3.0], [3]).unwrap();
    let t2 = Tensor::<f32, 1>::new(vec![4.0, 5.0], [2]).unwrap();

    let mut tensors = HashMap::new();
    tensors.insert("t1".to_string(), t1);
    tensors.insert("t2".to_string(), t2);

    save_checkpoint(&file_path, &tensors).unwrap();

    let loaded_tensors = load_checkpoint::<_, f32>(&file_path).unwrap();

    assert_eq!(loaded_tensors.len(), 2);

    let l1 = loaded_tensors.get("t1").unwrap();
    assert_eq!(l1.data(), &[1.0, 2.0, 3.0]);

    let l2 = loaded_tensors.get("t2").unwrap();
    assert_eq!(l2.data(), &[4.0, 5.0]);
}

#[test]
fn test_optimizer_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("optimizer.safetensors");

    // 1. Create and update optimizer
    let mut adam = AdamW::<f32>::new(0.001);
    let mut param = Tensor::new(vec![1.0], [1]).unwrap();
    let grad = Tensor::new(vec![0.1], [1]).unwrap();

    // Update to change state (m, v, step)
    adam.update(vec![&mut param], vec![&grad], 0).unwrap();

    // 2. Save state
    let state = adam.state_dict();
    save_checkpoint(&file_path, &state).unwrap();

    // 3. Load into new optimizer
    let mut adam_new = AdamW::<f32>::new(0.001);
    let loaded_state = load_checkpoint::<_, f32>(&file_path).unwrap();
    adam_new.load_state_dict(&loaded_state).unwrap();

    // 4. Verify state matches
    let state_old = adam.state_dict();
    let state_new = adam_new.state_dict();

    assert_eq!(state_old.len(), state_new.len());

    for (k, v) in state_old {
        let v_new = state_new.get(&k).unwrap();
        assert_eq!(v.data(), v_new.data());
    }
}
