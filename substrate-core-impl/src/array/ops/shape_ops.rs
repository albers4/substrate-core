// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{memory_order::MemoryOrder, ops::ShapeOps, ops::AccessOps, ArrayLike};

use crate::{Array, array::{error::ArrayError, utils::{compute_strides, unravel_index}}};

impl ShapeOps for Array<f64, Vec<f64>> {
    type Output = Self;
    type Error = ArrayError;
    
    fn reshape(self, new_shape: &[usize]) -> Result<Self, Self::Error> {
        if new_shape.is_empty() {
            return Err(ArrayError::EmptyShape);
        }
        if new_shape.contains(&0) {
            return Err(ArrayError::InvalidShapeDimension);
        }
        if new_shape.iter().product::<usize>() != self.length() {
            return Err(ArrayError::ReshapeSizeMismatch);
        }
        if !self.is_contiguous() {
            return Err(ArrayError::NotContiguous);
        }

        let strides = compute_strides(new_shape, self.order);
        Ok(Array {
            storage: self.storage,
            shape: new_shape.to_vec(),
            strides,
            offset: self.offset,
            order: self.order,
        })
    }
    
    fn reshape_copy(&self, new_shape: &[usize]) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn into_shape(self, new_shape: &[usize]) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn to_row_major(self) -> Result<Self, Self::Error> {
        if self.is_canonical(MemoryOrder::RowMajor) {
            return Ok(self);
        }

        let mut row_major_storage = vec![0.0f64; self.storage_length()];

        for (i, dst) in row_major_storage.iter_mut().enumerate() {
            let row_indices = unravel_index(i, self.shape(), MemoryOrder::RowMajor)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            let src_index: usize = self
                .physical_from_indices(&row_indices)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            *dst = *self.get_flat(src_index)?;
        }

        Ok(Array {
            storage: row_major_storage,
            shape: self.shape.to_vec(),
            strides: compute_strides(self.shape(), MemoryOrder::RowMajor),
            offset: 0,
            order: MemoryOrder::RowMajor,
        })
    }

    fn to_column_major(self) -> Result<Self, Self::Error> {
        if self.is_canonical(MemoryOrder::ColumnMajor) {
            return Ok(self);
        }

        let mut column_major_storage = vec![0.0f64; self.storage_length()];

        for (i, dst) in column_major_storage.iter_mut().enumerate() {
            let column_indices = unravel_index(i, self.shape(), MemoryOrder::ColumnMajor)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            let src_index: usize = self
                .physical_from_indices(&column_indices)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            *dst = *self.get_flat(src_index)?;
        }

        Ok(Array {
            storage: column_major_storage,
            shape: self.shape.to_vec(),
            strides: compute_strides(self.shape(), MemoryOrder::ColumnMajor),
            offset: 0,
            order: MemoryOrder::ColumnMajor,
        })
    }
    
    fn flatten(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn squeeze(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn unsqueeze(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn broadcast_to(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn concatenate(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn stack(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn split(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn roll(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn pad(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn tile(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

}