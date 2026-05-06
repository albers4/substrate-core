// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#[cfg(all(feature = "parallel", feature = "simd"))]
use {
    crate::{
        Array,
        array::{ArrayView, error::ArrayError, utils::compute_strides},
    },
    rayon::prelude::*,
    substrate_core_spec::array::{
        ArrayLike,
        memory_order::MemoryOrder,
        ops::{AccessOps, ConvertOps, LinearAlgebraOps, ShapeOps},
    },
};

#[cfg(all(feature = "parallel", feature = "simd"))]
impl<'a> ArrayView<'a, f64> {
    pub fn matmul_parallel_simd(
        &self,
        other: &ArrayView<'_, f64>,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let a_row = self.to_row_major_copy().unwrap();
        let b_row = other.to_column_major_copy().unwrap();
        let (m, _k) = (a_row.shape[0], a_row.shape[1]);
        let (_, n) = (b_row.shape()[0], b_row.shape()[1]);
        let out_shape = vec![m, n];
        let out_strides = compute_strides(&out_shape, MemoryOrder::RowMajor);

        let result = (0..m)
            .into_par_iter()
            .flat_map(|i| {
                let row = a_row.slice_by_range(0, i..i + 1).unwrap().squeeze().unwrap();
                (0..n)
                    .map(|j| {
                        let col = b_row
                            .slice_by_range(1, j..j + 1)
                            .unwrap()
                            .squeeze()
                            .unwrap();
                        row.view().dot(&col.view()).unwrap().to_scalar().unwrap()
                    })
                    .collect::<Vec<f64>>()
            })
            .collect();

        Ok(Array {
            storage: result,
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            order: MemoryOrder::RowMajor,
        })
    }
}
