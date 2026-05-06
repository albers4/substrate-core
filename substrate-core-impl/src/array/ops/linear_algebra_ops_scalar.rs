// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use crate::{
    Array,
    array::{ArrayView, error::ArrayError, utils::compute_strides},
};
use substrate_core_spec::array::{ArrayLike, memory_order::MemoryOrder, ops::{AccessOps, ShapeOps}};

impl<'a> ArrayView<'a, f64> {
    pub fn dot_scalar(
        &self,
        other: &ArrayView<'_, f64>,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let sum = self.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
        Ok(Array::from_scalar(sum))
    }

    pub fn matmul_scalar(
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

        for i in 0..m {
            for p in 0..k {
                let a_val = unsafe { a_row.get_unchecked(&[i, p]) };
                for j in 0..n {
                    result[i * n + j] += a_val * unsafe { b_row.get_unchecked(&[p, j]) };
                }
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
