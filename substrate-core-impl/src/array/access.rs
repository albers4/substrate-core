// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{
    ArrayAccess, ArrayAccessMut, index::ToIndex, memory_order::MemoryOrder,
};

use crate::array::{Array, error::ArrayError, utils::traversal_iters};
use core::fmt::Debug;

impl ArrayAccess for Array<f64, Vec<f64>> {
    type Item = f64;
    type Error = ArrayError;

    fn get_flat(&self, index: impl ToIndex) -> Result<&Self::Item, Self::Error> {
        let idx = index
            .to_index()
            .map_err(|_| ArrayError::IndexConversionError)?;
        let phys_idx = self.physical_from_logical_flat(idx)?;
        Ok(&self.storage.as_slice()[phys_idx])
    }

    fn get(&self, indices: &[impl ToIndex]) -> Result<&Self::Item, Self::Error> {
        self.physical_from_indices(indices)
            .map_or(Err(ArrayError::IndexOutOfBounds), |index| {
                Ok(&self.storage.as_slice()[index])
            })
    }

    unsafe fn get_unchecked<I: ToIndex>(&self, indices: &[I]) -> &Self::Item
    where
        I::Error: Debug,
    {
        let mut index: usize = self.offset();
        for (i, idx) in indices.iter().enumerate() {
            let dim = idx.to_index().unwrap();
            index += dim * self.strides[i];
        }
        unsafe { &*self.storage.as_ptr().add(index) }
    }

    fn first(&self) -> Result<&Self::Item, Self::Error> {
        self.storage
            .as_slice()
            .first()
            .ok_or(ArrayError::EmptyArray)
    }

    fn last(&self) -> Result<&Self::Item, Self::Error> {
        self.storage.as_slice().last().ok_or(ArrayError::EmptyArray)
    }

    fn iter(&self) -> impl Iterator<Item = Self::Item> {
        let pairs: Vec<(usize, usize)> =
            traversal_iters(self.shape.to_vec(), self.strides.to_vec(), self.order);
        (0..self.size()).map(move |flat: usize| {
            let mut index: usize = self.offset();
            let mut temp: usize = flat;
            for &(dim, stride) in &pairs {
                index += (temp % dim) * stride;
                temp /= dim;
            }
            self.storage.as_slice()[index]
        })
    }

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
        match self.order {
            MemoryOrder::RowMajor => self.strides.last() == Some(&1),
            MemoryOrder::ColumnMajor => self.strides.first() == Some(&1),
        }
    }

    fn is_canonical(&self, order: MemoryOrder) -> bool {
        self.order() == order && self.is_contiguous() && self.offset() == 0
    }

    fn to_scalar(&self) -> Result<Self::Item, Self::Error> {
        if self.length() != 1 {
            return Err(ArrayError::ArrayNotAScalar);
        }

        if let Ok(scalar) = self.get_flat(0) {
            Ok(*scalar)
        } else {
            Err(ArrayError::ArrayNotAScalar)
        }
    }

    fn to_vec(&self) -> Vec<Self::Item> {
        self.storage.as_slice().to_vec()
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

impl ArrayAccessMut for Array<f64, Vec<f64>> {
    fn get_flat_mut(&mut self, index: impl ToIndex) -> Result<&mut Self::Item, Self::Error> {
        let idx = index
            .to_index()
            .map_err(|_| ArrayError::IndexConversionError)?;
        let phys_idx = self.physical_from_logical_flat(idx)?;
        Ok(&mut self.storage.as_mut_slice()[phys_idx])
    }

    fn get_mut(&mut self, indices: &[impl ToIndex]) -> Result<&mut Self::Item, Self::Error> {
        self.physical_from_indices(indices)
            .map_or(Err(ArrayError::IndexOutOfBounds), |index| {
                Ok(&mut self.storage.as_mut_slice()[index])
            })
    }

    unsafe fn get_unchecked_mut<I: ToIndex>(&mut self, indices: &[I]) -> &mut Self::Item
    where
        I::Error: Debug,
    {
        let mut index: usize = self.offset();
        for (i, idx) in indices.iter().enumerate() {
            let dim = idx.to_index().unwrap();
            index += dim * self.strides[i];
        }
        unsafe { &mut *self.storage.as_mut_ptr().add(index) }
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item> {
        let len = self.size();
        let offset = self.offset();
        let shape = self.shape.clone();
        let strides = self.strides.clone();
        let order = self.order;
        let ptr = self.storage.as_mut_ptr();
        let pairs: Vec<(usize, usize)> = traversal_iters(shape, strides, order);

        (0..len).map(move |flat: usize| {
            let mut index: usize = offset;
            let mut temp: usize = flat;
            for &(dim, stride) in &pairs {
                index += (temp % dim) * stride;
                temp /= dim;
            }
            unsafe { &mut *ptr.add(index) }
        })
    }
}
