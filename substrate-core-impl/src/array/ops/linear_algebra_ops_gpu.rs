// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#[cfg(feature = "gpu")]
use kaio::runtime::KaioDevice;
#[cfg(feature = "gpu")]
use kaio_ops::matmul;
#[cfg(feature = "gpu")]
use substrate_core_spec::array::{
    ArrayLike,
    memory_order::MemoryOrder,
    ops::{AccessOps, ShapeOps},
};

#[cfg(feature = "gpu")]
use crate::{
    Array,
    array::{ArrayView, error::ArrayError, utils::compute_strides},
};

#[cfg(feature = "gpu")]
impl<'a> ArrayView<'a, f64> {
    pub fn matmul_gpu(
        &self,
        other: &ArrayView<'_, f64>,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let device = KaioDevice::new(0).unwrap();
        let a_row = self.to_row_major_copy().unwrap();
        let b_row = other.to_row_major_copy().unwrap();
        let (m, k) = (a_row.shape[0], a_row.shape[1]);
        let (_, n) = (b_row.shape()[0], b_row.shape()[1]);
        let out_shape = vec![m, n];
        let out_strides = compute_strides(&out_shape, MemoryOrder::RowMajor);

        let a_host = a_row.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let b_host = b_row.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let a = device.alloc_from(&a_host).unwrap();
        let b = device.alloc_from(&b_host).unwrap();
        let mut c = device.alloc_zeros::<f32>((m * n) as usize).unwrap();

        matmul(&device, &a, &b, &mut c, m as u32, n as u32, k as u32).unwrap();

        let result = c
            .to_host(&device)
            .unwrap()
            .iter()
            .map(|&x| x as f64)
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
