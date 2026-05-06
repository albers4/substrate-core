// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{
    ArrayLike, index::ToIndex, memory_order::MemoryOrder, number::Number, storage::Storage,
};

use crate::array::{error::ArrayError, utils::compute_strides};

pub struct Array<T: Number, S: Storage<Item = T>> {
    pub(crate) storage: S,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
    pub(crate) order: MemoryOrder,
}

pub type OwnedArray<T> = Array<T, Vec<T>>;

impl Array<f64, Vec<f64>> {
    pub fn from_scalar(scalar: f64) -> Self {
        Self {
            storage: vec![scalar],
            shape: vec![],
            strides: vec![],
            offset: 0,
            order: Default::default(),
        }
    }

    pub fn from_scalar_with_shape(scalar: f64, shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Self {
            storage: vec![scalar; size],
            shape: shape.to_vec(),
            strides: compute_strides(shape, Default::default()),
            offset: 0,
            order: Default::default(),
        }
    }

    pub fn from_vec_with_shape(data: Vec<f64>, shape: &[usize]) -> Result<Self, ArrayError> {
        if shape.is_empty() {
            return Err(ArrayError::EmptyShape);
        }
        if data.len() != shape.iter().product::<usize>() {
            return Err(ArrayError::DimensionMismatch);
        }

        Ok(Self {
            storage: data,
            shape: shape.to_vec(),
            strides: compute_strides(shape, Default::default()),
            offset: 0,
            order: Default::default(),
        })
    }
}

impl ArrayLike for Array<f64, Vec<f64>> {
    type Item = f64;
    type Error = ArrayError;

    fn length(&self) -> usize {
        self.storage_length() - self.offset
    }

    fn storage_length(&self) -> usize {
        self.storage.as_slice().len()
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
        self.strides == compute_strides(&self.shape, self.order)
    }

    fn is_canonical(&self, order: MemoryOrder) -> bool {
        self.order == order && self.is_contiguous() && self.offset == 0
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
            return Ok(self.offset() + index);
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
        debug_assert_eq!(remainder, 0, "Flat index decomposition failed");

        let mut phys = self.offset();
        for (d, coord) in coords.iter_mut().enumerate() {
            phys += *coord * self.strides[d];
        }
        Ok(phys)
    }
}
