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
