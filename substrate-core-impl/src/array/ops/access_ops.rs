// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{
    index::ToIndex, memory_order::MemoryOrder, ops::AccessOps, ArrayLike
};

use crate::array::{Array, error::ArrayError, utils::traversal_iters};
use core::fmt::Debug;

impl AccessOps for Array<f64, Vec<f64>> {
    type Item = f64;
    type Output = Self;
    type Error = ArrayError;

    fn get(&self, indices: &[impl ToIndex]) -> Result<&Self::Item, Self::Error> {
        self.physical_from_indices(indices)
            .map_or(Err(ArrayError::IndexOutOfBounds), |index| {
                Ok(&self.storage.as_slice()[index])
            })
    }

    fn slice_by_indices(&self, indices: &[impl ToIndex]) -> Result<Self, Self::Error> {
        if indices.len() != self.ndim() {
            return Err(ArrayError::DimensionMismatch);
        }

        let mut new_offset = self.offset;
        for (axis, idx) in indices.iter().enumerate() {
            let dim = idx
                .to_index()
                .map_err(|_| ArrayError::IndexConversionError)?;
            if dim >= self.shape[axis] {
                return Err(ArrayError::IndexOutOfBounds);
            }
            new_offset += dim * self.strides[axis];
        }

        Ok(Array {
            storage: self.storage.clone(),
            shape: vec![],
            strides: vec![],
            offset: new_offset,
            order: self.order,
        })
    }

    fn slice_by_range(
        &self,
        axis: usize,
        range: std::ops::Range<usize>,
    ) -> Result<Self, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }
        if range.start >= range.end || range.end > self.shape[axis] {
            return Err(ArrayError::InvalidSlice);
        }

        let mut new_shape = self.shape.clone();
        new_shape[axis] = range.end - range.start;
        let new_offset = self.offset + range.start * self.strides[axis];

        Ok(Array {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
            order: self.order,
        })
    }

    fn slice_by_stride(
        &self,
        axis: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Self, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }
        if step == 0 {
            return Err(ArrayError::InvalidSlice);
        }
        if start >= end || end > self.shape[axis] {
            return Err(ArrayError::InvalidSlice);
        }

        //let len = (end - start + step - 1) / step;
        let len = (end - start).div_ceil(step);
        let mut new_shape = self.shape.clone();
        new_shape[axis] = len;
        let mut new_strides = self.strides.clone();
        new_strides[axis] = self.strides[axis] * step;
        let new_offset = self.offset + start * self.strides[axis];

        Ok(Array {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
            order: self.order,
        })
    }

    fn select(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn take(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn gather(&self) -> Result<Self::Output, Self::Error> {
        todo!()
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

    fn get_flat(&self, index: impl ToIndex) -> Result<&Self::Item, Self::Error> {
        let idx = index
            .to_index()
            .map_err(|_| ArrayError::IndexConversionError)?;
        let phys_idx = self.physical_from_logical_flat(idx)?;
        Ok(&self.storage.as_slice()[phys_idx])
    }
}