// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{
    ArrayLike, ArrayViewLike, index::ToIndex, memory_order::MemoryOrder, number::Number,
};

use crate::array::{Array, core::OwnedArray, error::ArrayError};

#[derive(Clone)]
pub struct ArrayView<'a, T: Number> {
    pub(crate) data: &'a [T],
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
    pub(crate) order: MemoryOrder,
}

impl<'a> ArrayLike for ArrayView<'a, f64> {
    type Item = f64;
    type Error = ArrayError;

    fn length(&self) -> usize {
        self.storage_length() - self.offset
    }

    fn storage_length(&self) -> usize {
        self.data.len()
    }

    fn size(&self) -> usize {
        self.shape.iter().product()
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn order(&self) -> MemoryOrder {
        self.order
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn is_empty(&self) -> bool {
        self.length() == 0
    }

    fn is_contiguous(&self) -> bool {
        match self.order {
            MemoryOrder::RowMajor => self.strides.last() == Some(&1),
            MemoryOrder::ColumnMajor => self.strides.first() == Some(&1),
        }
    }

    fn is_canonical(&self, order: MemoryOrder) -> bool {
        self.order() == order && self.is_contiguous() && self.offset() == 0
    }

    fn physical_from_indices(&self, indices: &[impl ToIndex]) -> Result<usize, Self::Error> {
        if indices.len() != self.ndim() {
            return Err(ArrayError::DimensionMismatch);
        }

        let mut index = self.offset();
        for (i, idx) in indices.iter().enumerate() {
            let dim = idx
                .to_index()
                .map_err(|_| ArrayError::IndexConversionError)?;
            if dim >= self.shape[i] {
                return Err(ArrayError::IndexOutOfBounds);
            }
            index += dim * self.strides[i];
        }
        Ok(index)
    }

    fn physical_from_logical_flat(&self, index: usize) -> Result<usize, Self::Error> {
        if index >= self.length() {
            return Err(ArrayError::IndexOutOfBounds);
        }

        if self.is_contiguous() {
            return Ok(self.offset + index);
        }

        let mut coords = vec![0; self.ndim()];
        let mut remainder = index;

        match self.order() {
            MemoryOrder::RowMajor => {
                for dim in (0..self.ndim()).rev() {
                    let dim_size = self.shape[dim];
                    coords[dim] = remainder % dim_size;
                    remainder /= dim_size;
                }
            }
            MemoryOrder::ColumnMajor => {
                for (dim, coord) in coords.iter_mut().enumerate() {
                    let dim_size = self.shape[dim];
                    *coord = remainder % dim_size;
                    remainder /= dim_size;
                }
            }
        }

        let mut phys = self.offset();
        for (d, coord) in coords.iter().enumerate() {
            phys += *coord * self.strides[d];
        }
        Ok(phys)
    }
}

impl<'a> ArrayViewLike for ArrayView<'a, f64> {
    type Owned = OwnedArray<f64>;

    fn into_owned(self) -> Self::Owned {
        Array {
            storage: self.data.to_vec(),
            shape: self.shape.to_vec(),
            strides: self.strides.to_vec(),
            offset: self.offset,
            order: self.order,
        }
    }
}
