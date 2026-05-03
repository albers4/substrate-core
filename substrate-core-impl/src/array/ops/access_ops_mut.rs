// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{index::ToIndex, ops::AccessOpsMut, ArrayLike};

use crate::{Array, array::{error::ArrayError, utils::traversal_iters}};
use core::fmt::Debug;

impl AccessOpsMut for Array<f64, Vec<f64>> {
    fn get_mut(&mut self, indices: &[impl ToIndex]) -> Result<&mut Self::Item, Self::Error> {
        self.physical_from_indices(indices)
            .map_or(Err(ArrayError::IndexOutOfBounds), |index| {
                Ok(&mut self.storage.as_mut_slice()[index])
            })
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
    
    fn set_flat(&mut self, index: impl ToIndex, value: Self::Item) -> Result<(), Self::Error> {
        let idx = index
            .to_index()
            .map_err(|_| ArrayError::IndexConversionError)?;
        if idx >= self.size() {
            return Err(ArrayError::IndexOutOfBounds);
        }

        let pairs = traversal_iters(self.shape.to_vec(), self.strides.to_vec(), self.order);
        let mut flat_index = self.offset();
        let mut temp = idx;
        for &(dim, stride) in &pairs {
            flat_index += (temp % dim) * stride;
            temp /= dim;
        }
        self.storage.as_mut_slice()[flat_index] = value;

        Ok(())
    }

    fn set(&mut self, indices: &[impl ToIndex], value: Self::Item) -> Result<(), Self::Error> {
        let index = self.physical_from_indices(indices)?;
        self.storage.as_mut_slice()[index] = value;
        Ok(())
    }

    unsafe fn set_unchecked(
        &mut self,
        indices: &[impl ToIndex],
        value: Self::Item,
    ) -> Result<(), Self::Error> {
        let mut index: usize = self.offset();
        for (i, idx) in indices.iter().enumerate() {
            let dim = idx.to_index().unwrap();
            index += dim * self.strides[i];
        }
        unsafe { *self.storage.as_mut_ptr().add(index) = value }
        Ok(())
    }
    
    fn get_flat_mut(&mut self, index: impl ToIndex) -> Result<&mut Self::Item, Self::Error> {
        let idx = index
            .to_index()
            .map_err(|_| ArrayError::IndexConversionError)?;
        let phys_idx = self.physical_from_logical_flat(idx)?;
        Ok(&mut self.storage.as_mut_slice()[phys_idx])
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
}