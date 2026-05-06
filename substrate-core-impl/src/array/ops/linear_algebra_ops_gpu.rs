// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#[cfg(feature = "gpu")]
use crate::{
    Array,
    array::{ArrayView, error::ArrayError},
};

#[cfg(feature = "gpu")]
impl<'a> ArrayView<'a, f64> {
    pub fn matmul_gpu(
        &self,
        other: &ArrayView<'_, f64>,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        todo!()
    }
}
