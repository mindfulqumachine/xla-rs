use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use xla_rs::tensor::ops::TensorOps;
use xla_rs::tensor::{ConstDevice, Tensor};

fn benchmark_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");

    // CPU Transpose Benchmark
    // Create a 100x100 tensor (10,000 elements)
    let data = vec![1.0f32; 100 * 100];
    let t_cpu = Tensor::<f32, 2>::new(data, [100, 100]).unwrap();

    group.bench_function("cpu_100x100", |b| {
        b.iter(|| {
            // We benchmark the transpose operation itself
            black_box(t_cpu.transpose().unwrap());
        })
    });

    // Const Transpose Benchmark
    // We define the tensor and its transpose as constants.
    // The "benchmark" here is just accessing the pre-computed result.
    // This demonstrates that the cost is paid at compile time.
    const T_CONST: Tensor<f32, 2, ConstDevice<10000>> = Tensor::new_const([1.0; 10000], [100, 100]);
    const T_CONST_T: Tensor<f32, 2, ConstDevice<10000>> = T_CONST.transpose();

    group.bench_function("const_100x100_access", |b| {
        b.iter(|| {
            // We benchmark accessing the already transposed tensor
            black_box(T_CONST_T);
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_transpose);
criterion_main!(benches);
