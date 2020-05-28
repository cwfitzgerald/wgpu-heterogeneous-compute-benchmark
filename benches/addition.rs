use async_std::task::block_on;
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use std::time::{Duration, Instant};
use wgpu_heterogeneous_compute_benchmark::{
    addition, addition_iterator, addition_rayon, addition_unchecked, GPUAddition, UploadStyle,
};
use zerocopy::AsBytes;

pub fn criterion_benchmark(c: &mut Criterion) {
    env_logger::init();

    let mut group = c.benchmark_group("addition");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for size in &[
        1_000, 10_000, 100_000, 1_000_000, 3_000_000, 10_000_000, 50_000_000,
    ] {
        let data = vec![1.0_f32; *size];
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("scalar safe", size), size, |b, &_size| {
            b.iter_batched(
                || (data.clone(), data.clone()),
                |(mut left, right)| {
                    addition(&mut left, &right);

                    black_box(
                        left.as_bytes()
                            .iter()
                            .fold(true, |left, v| left ^ (*v == 1)),
                    );
                },
                BatchSize::LargeInput,
            );
        });
        group.bench_with_input(
            BenchmarkId::new("scalar unsafe", size),
            size,
            |b, &_size| {
                b.iter_batched(
                    || (data.clone(), data.clone()),
                    |(mut left, right)| {
                        unsafe { addition_unchecked(&mut left, &right) }

                        black_box(
                            left.as_bytes()
                                .iter()
                                .fold(true, |left, v| left ^ (*v == 1)),
                        );
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("scalar iterator", size),
            size,
            |b, &_size| {
                b.iter_batched(
                    || (data.clone(), data.clone()),
                    |(mut left, right)| {
                        addition_iterator(&mut left, &right);

                        black_box(
                            left.as_bytes()
                                .iter()
                                .fold(true, |left, v| left ^ (*v == 1)),
                        );
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        group.bench_with_input(BenchmarkId::new("scalar rayon", size), size, |b, &_size| {
            b.iter_batched(
                || (data.clone(), data.clone()),
                |(mut left, right)| {
                    addition_rayon(&mut left, &right);

                    black_box(
                        left.as_bytes()
                            .iter()
                            .fold(true, |left, v| left ^ (*v == 1)),
                    );
                },
                BatchSize::LargeInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("gpu mapping", size), size, |b, &_size| {
            b.iter_custom(|iterations| {
                let mut gpu = block_on(GPUAddition::new(UploadStyle::Mapping, *size));
                let mut duration = Duration::default();
                for _ in 0..iterations {
                    block_on(gpu.set_buffers(&data, &data));
                    let start = Instant::now();
                    let mapping = block_on(gpu.run(*size));
                    black_box(
                        mapping
                            .as_slice()
                            .iter()
                            .fold(true, |left, v| left ^ (*v == 1)),
                    );
                    duration += start.elapsed();
                    drop(mapping);
                }
                duration
            });
        });
        if *size <= 3_000_000 {
            group.bench_with_input(BenchmarkId::new("gpu staging", size), size, |b, &_size| {
                b.iter_custom(|iterations| {
                    let mut gpu = block_on(GPUAddition::new(UploadStyle::Staging, *size));
                    let mut duration = Duration::default();
                    for _ in 0..iterations {
                        block_on(gpu.set_buffers(&data, &data));
                        let start = Instant::now();
                        let mapping = block_on(gpu.run(*size));
                        black_box(
                            mapping
                                .as_slice()
                                .iter()
                                .fold(true, |left, v| left ^ (*v == 1)),
                        );
                        duration += start.elapsed();
                        drop(mapping);
                    }
                    duration
                });
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
