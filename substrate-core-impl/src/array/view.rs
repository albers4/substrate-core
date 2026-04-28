// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{
    ArrayAccess, ArrayViewLike, index::ToIndex, memory_order::MemoryOrder, number::Number,
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

        Ok(ArrayView {
            data: self.data,
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

        Ok(ArrayView {
            data: self.data,
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

        Ok(ArrayView {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
            order: self.order,
        })
    }
}
