use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use xla_rs_kernels::{cpu_matmul, cpu_transpose};

fn benchmark_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    let sizes = [64, 128, 256, 512];

    for &size in &sizes {
        let m = size;
        let k = size;
        let n = size;
        let lhs_shape = [m, k];
        let rhs_shape = [k, n];
        let lhs_data = vec![1.0f32; m * k];
        let rhs_data = vec![1.0f32; k * n];

        group.bench_function(format!("{}x{}", size, size), |b| {
            b.iter(|| {
                cpu_matmul(
                    black_box(&lhs_data),
                    black_box(&rhs_data),
                    black_box(&lhs_shape),
                    black_box(&rhs_shape),
                )
                .unwrap()
            })
        });
    }
    group.finish();
}

fn benchmark_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");
    let sizes = [128, 512, 1024, 2048];

    for &size in &sizes {
        let m = size;
        let n = size;
        let shape = [m, n];
        let data = vec![1.0f32; m * n];

        group.bench_function(format!("{}x{}", size, size), |b| {
            b.iter(|| cpu_transpose(black_box(&data), black_box(&shape)).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_matmul, benchmark_transpose);
criterion_main!(benches);
