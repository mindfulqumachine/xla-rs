//! Checkpointing and Serialization.
//!
//! This module provides functions to save and load model checkpoints using the `safetensors` format.

use crate::tensor::{Cpu, Result, Tensor, TensorElem};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

/// Saves a map of tensors to a safetensors file.
pub fn save_checkpoint<P: AsRef<Path>, T: TensorElem>(
    path: P,
    tensors: &HashMap<String, Tensor<T, 1, Cpu>>, // We assume flattened tensors for generic storage
) -> Result<()> {
    // Convert our Tensors to safetensors::TensorView
    // This requires T to be supported by safetensors (f32, f16, bf16, u8, i8, i32, i64, f64)
    // TensorElem usually covers these.

    // Note: safetensors expects a HashMap<String, TensorView>.
    // TensorView holds a reference to the data.

    let mut views = HashMap::new();

    // We need to keep the data alive while creating views.
    // The `tensors` argument provides references to data.

    for (name, tensor) in tensors {
        let dtype = match T::dtype_name() {
            "f32" => Dtype::F32,
            "f64" => Dtype::F64,
            // Add others as needed
            _ => Dtype::F32, // Fallback or panic?
        };

        let shape: Vec<usize> = tensor.shape().to_vec();

        // Unsafe cast to u8 slice for safetensors
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                tensor.data().as_ptr() as *const u8,
                std::mem::size_of_val(tensor.data()),
            )
        };

        let view = TensorView::new(dtype, shape, data_bytes).map_err(|e| {
            crate::tensor::TensorError::Unknown(format!("Safetensors error: {:?}", e))
        })?;

        views.insert(name.clone(), view);
    }

    let metadata: Option<HashMap<String, String>> = None;
    safetensors::serialize_to_file(&views, metadata, path.as_ref()).map_err(|e| {
        crate::tensor::TensorError::Unknown(format!("Failed to save checkpoint: {:?}", e))
    })?;

    Ok(())
}

/// Loads a map of tensors from a safetensors file.
pub fn load_checkpoint<P: AsRef<Path>, T: TensorElem>(
    path: P,
) -> Result<HashMap<String, Tensor<T, 1, Cpu>>> {
    let file_content = std::fs::read(path).map_err(|e| {
        crate::tensor::TensorError::Unknown(format!("Failed to read file: {:?}", e))
    })?;

    let safetensors = SafeTensors::deserialize(&file_content).map_err(|e| {
        crate::tensor::TensorError::Unknown(format!("Failed to deserialize safetensors: {:?}", e))
    })?;

    let mut tensors = HashMap::new();

    for (name, view) in safetensors.tensors() {
        // Check dtype
        // For now assume T matches file content or we cast?
        // Safetensors is strict.

        let data_u8 = view.data();

        // Convert u8 slice back to Vec<T>
        // This copies the data.
        let element_count = data_u8.len() / std::mem::size_of::<T>();
        let mut data_t = Vec::with_capacity(element_count);

        unsafe {
            let ptr = data_u8.as_ptr() as *const T;
            for i in 0..element_count {
                data_t.push(*ptr.add(i));
            }
        }

        // Shape
        let _shape = view.shape().to_vec();
        // We load as Rank 1 (flattened) because our HashMap signature says so.
        // The caller (Model/Optimizer) is responsible for reshaping.
        // Or we can return a custom struct that holds shape info.
        // But `Tensor<T, 1, Cpu>` is strictly rank 1.
        // If we want to preserve shape, we might need a wrapper enum or `AnyTensor`.
        // For now, let's return flattened tensors and let the loader reshape.

        let tensor = Tensor::new(data_t, [element_count]).unwrap();
        tensors.insert(name, tensor);
    }

    Ok(tensors)
}
