// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use pprof::criterion::{Output, PProfProfiler};
use substrate_core_impl::Array;
use substrate_core_spec::array::{ArrayAccess, ArrayLike};

fn array_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_creation");

    group.bench_function("scalar", |bench| {
        let arr = Array::from_vec(black_box(vec![0.0, 1.0, 2.0, 3.0, 4.0]));

        bench.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..arr.length() {
                sum += black_box(i as f64);
            }
            black_box(sum)
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = array_benchmark
}
criterion_main!(benches);
