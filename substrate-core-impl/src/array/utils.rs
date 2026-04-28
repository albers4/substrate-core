// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::memory_order::MemoryOrder;

use crate::array::error::ArrayError;

pub fn compute_strides(shape: &[usize], order: MemoryOrder) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];

    match order {
        MemoryOrder::RowMajor => {
            for i in (0..shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        MemoryOrder::ColumnMajor => {
            for i in 1..shape.len() {
                strides[i] = strides[i - 1] * shape[i - 1];
            }
        }
    }

    strides
}

pub fn traversal_iters(
    shape: Vec<usize>,
    strides: Vec<usize>,
    order: MemoryOrder,
) -> Vec<(usize, usize)> {
    let traversal: Vec<(usize, usize)> = if matches!(order, MemoryOrder::RowMajor) {
        shape
            .iter()
            .rev()
            .copied()
            .zip(strides.iter().rev().copied())
            .collect()
    } else {
        shape.iter().copied().zip(strides.iter().copied()).collect()
    };
    traversal
}

pub fn unravel_index(
    index: usize,
    shape: &[usize],
    order: MemoryOrder,
) -> Result<Vec<usize>, ArrayError> {
    if index >= shape.iter().product() {
        return Err(ArrayError::IndexOutOfBounds);
    }
    if shape.is_empty() {
        return Err(ArrayError::EmptyShape);
    }

    let mut indices: Vec<usize> = vec![0; shape.len()];

    match order {
        MemoryOrder::RowMajor => {
            let mut flat_index: usize = index;
            for i in (0..shape.len()).rev() {
                indices[i] = flat_index % shape[i];
                flat_index /= shape[i];
            }
        }
        MemoryOrder::ColumnMajor => {
            let mut flat_index: usize = index;
            for i in 0..shape.len() {
                indices[i] = flat_index % shape[i];
                flat_index /= shape[i];
            }
        }
    }

    Ok(indices)
}

pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>, ArrayError> {
    let mut result: Vec<usize> = Vec::new();
    let mut i: usize = shape1.len();
    let mut j: usize = shape2.len();

    while i > 0 || j > 0 {
        let d1: usize = if i > 0 { shape1[i - 1] } else { 1 };
        let d2: usize = if j > 0 { shape2[j - 1] } else { 1 };

        if d1 != 1 && d2 != 1 && d1 != d2 {
            return Err(ArrayError::IncompatibleShapes);
        }

        result.push(d1.max(d2));
        i = i.saturating_sub(1);
        j = j.saturating_sub(1);
    }

    Ok(result.into_iter().rev().collect())
}

pub fn broadcast_strides(orig_shape: &[usize], orig_strides: &[usize], target_shape: &[usize]) -> Result<Vec<usize>, ArrayError> {
    let mut new_strides: Vec<usize> = vec![0; target_shape.len()];

    let orig_ndim: usize = orig_shape.len();
    let target_ndim: usize = target_shape.len();

    for i in 0..orig_ndim {
        let target_dim = target_ndim - orig_ndim + i;
        let orig_dim = i;

        if orig_shape[orig_dim] == 1 {
            new_strides[target_dim] = 0;
        } else {
            new_strides[target_dim] = orig_strides[orig_dim];
        }
    }

    Ok(new_strides)
}