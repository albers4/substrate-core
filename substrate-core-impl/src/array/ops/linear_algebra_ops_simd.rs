// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#[cfg(feature = "simd")]
use {
    crate::{
        Array,
        array::{ArrayView, error::ArrayError, utils::compute_strides},
    },
    substrate_core_spec::array::{
        ArrayLike,
        memory_order::MemoryOrder,
        ops::{AccessOps, ConvertOps, LinearAlgebraOps, ShapeOps},
    },
};

#[cfg(feature = "simd")]
impl<'a> ArrayView<'a, f64> {
    pub fn dot_simd(&self, other: &ArrayView<'_, f64>) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let sum = self.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
        Ok(Array::from_scalar(sum))
    }

    pub fn matmul_simd(
        &self,
        other: &ArrayView<'_, f64>,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let (m, _k) = (self.shape[0], self.shape[1]);
        let (_, n) = (other.shape()[0], other.shape()[1]);
        let out_shape = vec![m, n];
        let out_strides = compute_strides(&out_shape, MemoryOrder::RowMajor);

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            let row = self.slice_by_range(0, i..i + 1)?.squeeze()?;
            for j in 0..n {
                let col = other.slice_by_range(1, j..j + 1)?.squeeze()?;
                let dot_val = row.view().dot(&col.view())?.to_scalar()?;
                result[i * n + j] = dot_val;
            }
        }

        Ok(Array {
            storage: result,
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            order: MemoryOrder::RowMajor,
        })
    }
}
