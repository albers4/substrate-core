// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{ArrayLike, index::ToIndex, ops::AccessOpsMut};

use crate::{
    Array,
    array::{error::ArrayError, utils::traversal_iters},
};
use core::fmt::Debug;

impl AccessOpsMut for Array<f64, Vec<f64>> {
    /// Returns a mutable reference to the element at the given multi‑dimensional indices.
    ///
    /// # Arguments
    /// * `indices` - A slice of indices, one per axis, each convertible to `usize`.
    ///
    /// # Returns
    /// `Ok(&mut Self::Item)` if all indices are within bounds and the number of indices matches
    /// the dimension of the array, otherwise `Err(ArrayError::DimensionMismatch)` or
    /// `Err(ArrayError::IndexOutOfBounds)`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, AccessOpsMut, ConvertOps};
    ///
    /// let mut a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// *a.get_mut(&[1, 1]).unwrap() = 99.0;
    /// assert_eq!(a.to_vec(), vec![1.0, 2.0, 3.0, 99.0]);
    /// ```
    fn get_mut(&mut self, indices: &[impl ToIndex]) -> Result<&mut Self::Item, Self::Error> {
        self.physical_from_indices(indices)
            .map_or(Err(ArrayError::IndexOutOfBounds), |index| {
                Ok(&mut self.storage.as_mut_slice()[index])
            })
    }

    /// Returns a mutable iterator over the elements of the array in logical index order.
    ///
    /// The iteration respects the array’s memory layout (row‑major or column‑major) and
    /// works correctly for non‑contiguous views. The order of elements corresponds to the
    /// logical flat indexing (0 ... `size()-1`) according to the array’s `MemoryOrder`.
    ///
    /// # Returns
    /// An iterator yielding mutable references (`&mut Self::Item`) to each element.
    ///
    /// # Safety
    /// The method internally uses unsafe pointer arithmetic, but the indices are computed
    /// correctly from the array’s shape, strides, and offset, so the behaviour is safe
    /// as long as the array’s underlying storage remains valid for the iterator’s lifetime.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, AccessOpsMut, ConvertOps};
    ///
    /// let mut a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// for x in a.iter_mut() {
    ///     *x *= 2.0;
    /// }
    /// assert_eq!(a.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    /// ```
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

    /// Sets the element at the given flat logical index to the specified value.
    ///
    /// The flat index is interpreted in the logical order (row‑major or column‑major)
    /// according to the array’s `MemoryOrder`. The index is mapped to the physical
    /// position using the array’s strides and offset.
    ///
    /// # Arguments
    /// * `index` - Flat logical index (0 ... `size()-1`) convertible to `usize` via the `ToIndex` trait.
    /// * `value` - The new value to assign.
    ///
    /// # Returns
    /// `Ok(())` on success, or an error if the index is out of bounds or conversion fails.
    ///
    /// # Errors
    /// * `ArrayError::IndexConversionError` – if `index.to_index()` fails.
    /// * `ArrayError::IndexOutOfBounds` – if `idx >= self.size()`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps, AccessOpsMut};
    ///
    /// let mut a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    /// a.set_flat(2, 99.0).unwrap();
    /// assert_eq!(a.to_vec(), vec![1.0, 2.0, 99.0, 4.0]);
    /// ```
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

    /// Sets the element at the given multi‑dimensional indices to the specified value.
    ///
    /// # Arguments
    /// * `indices` - A slice of indices, one per axis, each convertible to `usize`.
    /// * `value` - The new value to assign.
    ///
    /// # Returns
    /// `Ok(())` on success, or an error if indices are out of bounds or dimension mismatch.
    ///
    /// # Errors
    /// * `ArrayError::DimensionMismatch` – if the number of indices does not match `ndim()`.
    /// * `ArrayError::IndexOutOfBounds` – if any index is outside the corresponding shape dimension.
    /// * `ArrayError::IndexConversionError` – if any index fails conversion to `usize`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps, AccessOpsMut};
    ///
    /// let mut a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// a.set(&[1, 1], 99.0).unwrap();
    /// assert_eq!(a.to_vec(), vec![1.0, 2.0, 3.0, 99.0]);
    /// ```
    fn set(&mut self, indices: &[impl ToIndex], value: Self::Item) -> Result<(), Self::Error> {
        let index = self.physical_from_indices(indices)?;
        self.storage.as_mut_slice()[index] = value;
        Ok(())
    }

    /// Sets the element at the given multi‑dimensional indices without bounds checking.
    ///
    /// # Safety
    /// - `indices` must contain exactly `ndim()` elements.
    /// - Each index must be strictly less than the corresponding shape dimension.
    /// - The computed physical index must be within the allocated storage.
    /// - The caller must ensure that no aliasing rules are violated (the reference is unique).
    ///
    /// # Arguments
    /// * `indices` - A slice of indices, one per axis, each convertible to `usize`.
    /// * `value` - The new value to assign.
    ///
    /// # Returns
    /// `Ok(())` on success. (No error checking is performed.)
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps, AccessOpsMut};
    ///
    /// let mut a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// unsafe { a.set_unchecked(&[1, 1], 99.0).unwrap() };
    /// assert_eq!(a.to_vec(), vec![1.0, 2.0, 3.0, 99.0]);
    /// ```
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

    /// Returns a mutable reference to the element at the given flat logical index.
    ///
    /// The flat index is interpreted in the logical order (row‑major or column‑major)
    /// according to the array’s `MemoryOrder`. For a contiguous array, this corresponds
    /// to the physical layout; for non‑contiguous arrays, the index is mapped to the
    /// correct physical position using strides.
    ///
    /// # Arguments
    /// * `index` - Flat index (0..len-1) convertible to `usize` via the `ToIndex` trait.
    ///
    /// # Returns
    /// `Ok(&mut Self::Item)` if the index is valid, otherwise `Err(ArrayError::IndexOutOfBounds)`
    /// or `Err(ArrayError::IndexConversionError)`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps, AccessOpsMut};
    ///
    /// let mut a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// *a.get_flat_mut(2).unwrap() = 99.0;
    /// assert_eq!(a.to_vec(), vec![1.0, 2.0, 99.0, 4.0]);
    /// ```
    fn get_flat_mut(&mut self, index: impl ToIndex) -> Result<&mut Self::Item, Self::Error> {
        let idx = index
            .to_index()
            .map_err(|_| ArrayError::IndexConversionError)?;
        let phys_idx = self.physical_from_logical_flat(idx)?;
        Ok(&mut self.storage.as_mut_slice()[phys_idx])
    }

    /// Returns a mutable reference to the element at the given multi‑dimensional indices without bounds checking.
    ///
    /// # Safety
    /// - `indices` must contain exactly `ndim()` elements.
    /// - Each index must be strictly less than the corresponding shape dimension.
    /// - The computed physical index must be within the allocated storage.
    /// - The caller must ensure that the returned reference does not alias any other mutable reference to the same element.
    ///
    /// # Arguments
    /// * `indices` - A slice of indices, one per axis, each convertible to `usize`.
    ///
    /// # Returns
    /// A mutable reference to the element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps, AccessOpsMut};
    ///
    /// let mut a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// unsafe {
    ///     *a.get_unchecked_mut(&[1, 1]) = 99.0;
    /// }
    /// assert_eq!(a.to_vec(), vec![1.0, 2.0, 3.0, 99.0]);
    /// ```
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
