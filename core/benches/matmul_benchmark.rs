use criterion::{Criterion, black_box, criterion_group, criterion_main};
use xla_rs::tensor::{ConstDevice, Tensor};

fn benchmark_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    // CPU Matmul Benchmark
    // 20x20 matrices (400 elements)
    let data_a = vec![1.0f32; 20 * 20];
    let data_b = vec![1.0f32; 20 * 20];
    let t_cpu_a = Tensor::<f32, 2>::new(data_a, [20, 20]).unwrap();
    let t_cpu_b = Tensor::<f32, 2>::new(data_b, [20, 20]).unwrap();

    group.bench_function("cpu_20x20", |b| {
        b.iter(|| {
            black_box(t_cpu_a.matmul(&t_cpu_b).unwrap());
        })
    });

    // Const Matmul Benchmark
    // 20x20 matrices
    const A: Tensor<f32, 2, ConstDevice<400>> = Tensor::new_const([1.0; 400], [20, 20]);
    const B: Tensor<f32, 2, ConstDevice<400>> = Tensor::new_const([1.0; 400], [20, 20]);
    const C: Tensor<f32, 2, ConstDevice<400>> = A.matmul(&B);

    group.bench_function("const_20x20_access", |b| {
        b.iter(|| {
            // Benchmark accessing the pre-computed result
            black_box(C);
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_matmul);
criterion_main!(benches);
