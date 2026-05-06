// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#[cfg(feature = "parallel")]
use {
    crate::{
        Array,
        array::{ArrayView, error::ArrayError, utils::compute_strides},
    },
    rayon::prelude::*,
    substrate_core_spec::array::{ArrayLike, memory_order::MemoryOrder, ops::{AccessOps, ShapeOps}},
};

#[cfg(feature = "parallel")]
impl<'a> ArrayView<'a, f64> {
    pub fn matmul_parallel(
        &self,
        other: &ArrayView<'_, f64>,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let a_row = self.to_row_major_copy().unwrap();
        let b_row = other.to_column_major_copy().unwrap();
        let (m, k) = (a_row.shape[0], a_row.shape[1]);
        let (_, n) = (b_row.shape()[0], b_row.shape()[1]);
        let out_shape = vec![m, n];
        let out_strides = compute_strides(&out_shape, MemoryOrder::RowMajor);

        let mut result = vec![0.0; m * n];

        result.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    let a = *a_row.get(&[i, p]).unwrap();
                    let b = *b_row.get(&[p, j]).unwrap();
                    sum += a * b;
                }
                row[j] = sum;
            }
        });

        Ok(Array {
            storage: result,
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            order: MemoryOrder::RowMajor,
        })
    }
}
