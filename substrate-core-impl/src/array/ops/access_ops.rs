// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{
    ArrayLike,
    index::ToIndex,
    ops::{AccessOps, ConvertOps},
};

use crate::array::{
    Array, ArrayView,
    error::ArrayError,
    utils::{traversal_iters, unravel_index},
};
use core::fmt::Debug;

impl<'a> AccessOps for ArrayView<'a, f64> {
    type Output = Array<f64, Vec<f64>>;
    type View<'b>
        = ArrayView<'b, f64>
    where
        Self: 'b;

    /// Returns a reference to the element at the given multi‑dimensional indices.
    ///
    /// # Arguments
    /// * `indices` - A slice of indices, one per axis, each convertible to `usize`.
    ///
    /// # Returns
    /// `Ok(&Self::Item)` if all indices are within bounds and the number of indices matches
    /// the dimension of the array, otherwise `Err(ArrayError::DimensionMismatch)` or
    /// `Err(ArrayError::IndexOutOfBounds)`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, AccessOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// assert_eq!(*a.get(&[1, 1]).unwrap(), 4.0);
    /// ```
    fn get(&self, indices: &[impl ToIndex]) -> Result<&Self::Item, Self::Error> {
        self.physical_from_indices(indices)
            .map_or(Err(ArrayError::IndexOutOfBounds), |index| {
                Ok(&self.data[index])
            })
    }

    /// Slices an `ArrayView<f64>` by indices.
    ///
    /// # Arguments
    /// * `indices` - List of indices to slice along
    ///
    /// # Returns
    /// `ArrayView<f64>` containing a single item
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::AccessOps;
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec_with_shape(vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), &[2, 3]).unwrap();
    /// let a_slice = a.slice_by_indices(&[1, 1]).unwrap();
    /// let res: f64 = *a_slice.first().unwrap();
    /// assert_eq!(a_slice.shape(), &[] as &[usize]);
    /// assert_eq!(a_slice.strides(), &[] as &[usize]);
    /// assert_eq!(res, 5.0);
    /// assert_eq!(a_slice.ndim(), 0);
    /// assert_eq!(a_slice.size(), 1);
    /// ```
    fn slice_by_indices(&self, indices: &[impl ToIndex]) -> Result<Self::View<'_>, Self::Error> {
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

    /// Slices an `ArrayView<f64>` by a range.
    ///
    /// # Arguments
    /// * `axis` - Axis to slice along
    /// * `range` - Range defining start and end
    ///
    /// # Returns
    /// `ArrayView<f64>`
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::AccessOps;
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec_with_shape(vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), &[2, 3]).unwrap();
    /// let a_slice = a.slice_by_range(1, 1..3).unwrap();
    /// assert_eq!(a_slice.shape(), [2, 2]);
    /// assert_eq!(a_slice.strides(), [3, 1]);
    /// assert_eq!(a_slice.iter().copied().collect::<Vec<_>>(), vec![2.0, 3.0, 5.0, 6.0]);
    /// assert_eq!(a_slice.ndim(), 2);
    /// assert_eq!(a_slice.size(), 4);
    /// assert_eq!(a_slice.offset(), a.offset() + 1 * a.strides()[1]);
    /// ```
    fn slice_by_range(
        &self,
        axis: usize,
        range: std::ops::Range<usize>,
    ) -> Result<Self::View<'_>, Self::Error> {
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

    /// Slices an `ArrayView<f64>` by a stride.
    ///
    /// # Arguments
    /// * `axis` - Axis to slice along
    /// * `start` - Start index in specified axis
    /// * `end` - End index in specified axis
    /// * `step` - Size of step along the axis between start and end
    ///
    /// # Returns
    /// `ArrayView<f64>`
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::AccessOps;
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec_with_shape(vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), &[2, 4]).unwrap();
    /// let a_slice = a.slice_by_stride(1, 0, 3, 2).unwrap();
    /// assert_eq!(a_slice.shape(), [2, 2]);
    /// assert_eq!(a_slice.strides(), [4, 2]);
    /// assert_eq!(a_slice.iter().copied().collect::<Vec<_>>(), vec![1.0, 3.0, 5.0, 7.0]);
    /// assert_eq!(a_slice.ndim(), 2);
    /// assert_eq!(a_slice.size(), 4);
    /// ```
    fn slice_by_stride(
        &self,
        axis: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Self::View<'_>, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }
        if step == 0 {
            return Err(ArrayError::InvalidSlice);
        }
        if start >= end || end > self.shape[axis] {
            return Err(ArrayError::InvalidSlice);
        }

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

    /// Returns a new array containing elements where the corresponding mask value is `1.0`.
    ///
    /// The mask must have the same shape as the view. Only elements for which the mask
    /// equals `1.0` are kept; all others are ignored. The result is a **1‑dimensional**
    /// array (flattened in logical order) containing the selected values.
    ///
    /// # Arguments
    /// * `mask` – An `ArrayView` of `f64` where `1.0` indicates “keep” and any other value
    ///   (including `0.0`) indicates “discard”.
    ///
    /// # Errors
    /// * `ArrayError::IncompatibleShapes` – if `self.shape() != mask.shape()`.
    /// * `ArrayError::IndexConversionError` – if any index conversion fails (unlikely for mask).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with shape `[n]`, where `n` is the number of `1.0` entries
    /// in the mask.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps, AccessOps};
    ///
    /// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let view = arr.view();
    /// let mask = Array::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0]);
    /// let selected = view.select(&mask.view()).unwrap();
    /// assert_eq!(selected.to_vec(), vec![1.0, 3.0, 5.0]);
    /// ```
    fn select(&self, mask: &ArrayView<'_, f64>) -> Result<Self::Output, Self::Error> {
        if self.shape() != mask.shape() {
            return Err(ArrayError::IncompatibleShapes);
        }

        let mut out_data = Vec::new();
        for (&val, &keep) in self.iter().zip(mask.iter()) {
            if keep == 1.0 {
                out_data.push(val);
            }
        }
        let out_shape = vec![out_data.len()];
        Array::from_vec_with_shape(out_data, &out_shape)
    }

    /// Returns a new array by taking elements at the given flat indices.
    ///
    /// The `indices` array is interpreted as a 1‑D sequence of flat indices (logical order).
    /// For each index, the corresponding element from the view is fetched and appended to
    /// the output. The output shape is `[len(indices)]`.
    ///
    /// # Arguments
    /// * `indices` – An `ArrayView` of `f64`, where each value is converted to `usize`
    ///   and used as a flat index into the view.
    ///
    /// # Errors
    /// * `ArrayError::IndexOutOfBounds` – if any index value is out of range for `self`.
    /// * `ArrayError::IndexConversionError` – if an index cannot be converted to `usize`.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same length as `indices`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps, AccessOps};
    ///
    /// let arr = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    /// let view = arr.view();
    /// let idx = Array::from_vec(vec![4.0, 0.0, 2.0]);
    /// let taken = view.take(&idx.view()).unwrap();
    /// assert_eq!(taken.to_vec(), vec![50.0, 10.0, 30.0]);
    /// ```
    fn take(&self, indices: &ArrayView<'_, f64>) -> Result<Self::Output, Self::Error> {
        let mut out_data = Vec::with_capacity(indices.length());

        for idx in indices.iter() {
            let val = *self.get_flat(*idx)?;
            out_data.push(val);
        }

        let out_shape = vec![out_data.len()];
        Array::from_vec_with_shape(out_data, &out_shape)
    }

    /// Gathers values from the view along a given axis using an index array.
    ///
    /// For each position in the output (which has the same shape as `indices`), the value is
    /// taken from the view at coordinates that are the same as the output coordinates, except
    /// along the specified `dim` where the coordinate is taken from the `indices` array.
    ///
    /// # Arguments
    /// * `dim` – The axis along which to gather (0‑based). Must be less than `self.ndim()`.
    /// * `indices` – An `ArrayView` of `f64` containing the indices (converted to `usize`)
    ///   along dimension `dim`. The shape of `indices` determines the output shape.
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `dim >= self.ndim()`.
    /// * `ArrayError::IndexOutOfBounds` – if any gathered index value is out of bounds for that axis.
    /// * `ArrayError::IndexConversionError` – if an index cannot be converted to `usize`.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape as `indices`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps, AccessOps, ShapeOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .reshape(&[2, 3])
    ///     .unwrap();
    /// let view = arr.view();
    /// // Indices: shape [2,2], each value picks column index
    /// let indices = Array::from_vec(vec![0.0, 2.0, 1.0, 0.0])
    ///     .reshape(&[2, 2])
    ///     .unwrap();
    /// let gathered = arr.view().gather(1, &indices.view()).unwrap();
    /// // Output shape [2,2]:
    /// // row0: col0=1, col2=3
    /// // row1: col1=5, col0=4
    /// assert_eq!(gathered.to_vec(), vec![1.0, 3.0, 5.0, 4.0]);
    /// assert_eq!(gathered.shape(), &[2, 2]);
    /// ```
    fn gather(
        &self,
        dim: impl ToIndex,
        indices: &ArrayView<'_, f64>,
    ) -> Result<Self::Output, Self::Error> {
        let dim = dim.to_index().unwrap();
        if dim >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }

        let out_shape = indices.shape().to_vec();
        let mut out_data = vec![0.0; indices.length()];
        for (out_flat, out_item) in out_data.iter_mut().enumerate() {
            let out_coords = unravel_index(out_flat, indices.shape(), indices.order())?;
            let mut src_coords = out_coords.clone();
            src_coords[dim] = *indices.get_flat(out_flat)? as usize;
            let src_flat = self.physical_from_indices(&src_coords)?;
            *out_item = self.data[src_flat];
        }

        Array::from_vec_with_shape(out_data, &out_shape)
    }

    /// Returns an iterator over the elements of the array view.
    ///
    /// The iterator traverses the elements in logical index order according to the
    /// memory order (row‑major or column‑major) of the view. This respects any
    /// non‑contiguous strides.
    ///
    /// # Returns
    /// An iterator yielding elements of type `Self::Item`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{AccessOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec_with_shape(vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), &[2, 4]).unwrap();
    /// let view = a.view();
    /// let collected: Vec<f64> = view.iter().copied().collect();
    /// assert_eq!(collected, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    ///
    /// // Non‑contiguous slice (every second column)
    /// let sliced = view.slice_by_stride(1, 0, 4, 2).unwrap();
    /// assert_eq!(sliced.iter().copied().collect::<Vec<_>>(), vec![1.0, 3.0, 5.0, 7.0]);
    /// ```
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        let pairs: Vec<(usize, usize)> =
            traversal_iters(self.shape.to_vec(), self.strides.to_vec(), self.order);
        (0..self.size()).map(move |flat: usize| {
            let mut index: usize = self.offset();
            let mut temp: usize = flat;
            for &(dim, stride) in &pairs {
                index += (temp % dim) * stride;
                temp /= dim;
            }
            &self.data[index]
        })
    }

    /// Returns a reference to the element at the given multi‑dimensional indices without bounds checking.
    ///
    /// # Safety
    /// - `indices` must contain exactly `ndim()` elements.
    /// - Each index must be strictly less than the corresponding shape dimension.
    /// - The computed physical index must be within the underlying data slice.
    /// - The caller must ensure that the resulting reference does not outlive the view or
    ///   that aliasing rules are not violated.
    ///
    /// # Arguments
    /// * `indices` - Multi‑dimensional index per axis.
    ///
    /// # Returns
    /// A reference to the element of type `&Self::Item`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{AccessOps, InitOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// let view = a.view();
    /// unsafe {
    ///     assert_eq!(*view.get_unchecked(&[1, 1]), 4.0);
    /// }
    /// ```
    unsafe fn get_unchecked<I: ToIndex>(&self, indices: &[I]) -> &Self::Item
    where
        I::Error: Debug,
    {
        let mut index: usize = self.offset();
        for (i, idx) in indices.iter().enumerate() {
            let dim = idx.to_index().unwrap();
            index += dim * self.strides[i];
        }
        unsafe { &*self.data.as_ptr().add(index) }
    }

    /// Returns a reference to the first element of the array view.
    ///
    /// The first element corresponds to the logical index where all axes are zero.
    /// This respects the view's offset and strides.
    ///
    /// # Returns
    /// `Ok(&Self::Item)` if the view is non‑empty, otherwise `Err(ArrayError::EmptyArray)`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{AccessOps, InitOps, ShapeOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// let view = a.view();
    /// assert_eq!(*view.first().unwrap(), 1.0);
    ///
    /// let empty = Array::from_vec(vec![]);
    /// assert!(empty.first().is_err());
    /// ```
    fn first(&self) -> Result<&Self::Item, Self::Error> {
        if self.length() == 0 {
            return Err(ArrayError::EmptyArray);
        }

        Ok(&self.data[self.offset])
    }

    /// Returns a reference to the last element of the array view.
    ///
    /// The last element corresponds to the logical index where each axis is at its maximum
    /// (shape[axis] - 1). The physical position is computed using the view's offset and strides.
    ///
    /// # Returns
    /// `Ok(&Self::Item)` if the view is non‑empty, otherwise `Err(ArrayError::EmptyArray)`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{AccessOps, InitOps, ShapeOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// assert_eq!(*a.last().unwrap(), 4.0);
    ///
    /// let empty = Array::from_vec(vec![]);
    /// assert!(empty.last().is_err());
    /// ```
    fn last(&self) -> Result<&Self::Item, Self::Error> {
        if self.length() == 0 {
            return Err(ArrayError::EmptyArray);
        }
        let mut index = self.offset;
        for d in 0..self.ndim() {
            index += (self.shape[d] - 1) * self.strides[d];
        }
        Ok(&self.data[index])
    }

    /// Returns a reference to the element at the given flat logical index.
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
    /// `Ok(&Self::Item)` if the index is valid, otherwise `Err(ArrayError::IndexOutOfBounds)`
    /// or `Err(ArrayError::IndexConversionError)`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, AccessOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// assert_eq!(*a.get_flat(2).unwrap(), 3.0);
    ///
    /// // Out‑of‑bounds
    /// assert!(a.get_flat(4).is_err());
    /// ```
    fn get_flat(&self, index: impl ToIndex) -> Result<&Self::Item, Self::Error> {
        let idx = index
            .to_index()
            .map_err(|_| ArrayError::IndexConversionError)?;
        let phys_idx = self.physical_from_logical_flat(idx)?;
        Ok(&self.data[phys_idx])
    }

    /// Returns the raw underlying data slice of the array view.
    ///
    /// The returned slice includes all elements of the allocated storage, including
    /// those that may lie outside the logical view (e.g., due to a non‑zero offset
    /// or strided access). For the logical view elements, use `iter()` instead.
    ///
    /// # Returns
    /// A slice `&[Self::Item]` representing the underlying buffer.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{ConvertOps, InitOps, AccessOps, ShapeOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// let view = a.view();
    /// assert_eq!(view.data(), &[1.0, 2.0, 3.0, 4.0]);
    ///
    /// // After slicing, the data slice still contains the full buffer
    /// let sliced = view.slice_by_range(0, 0..1).unwrap();
    /// assert_eq!(sliced.data(), &[1.0, 2.0, 3.0, 4.0]); // not just [1.0, 2.0]
    /// ```
    fn data(&self) -> &[Self::Item] {
        self.data
    }
}

impl AccessOps for Array<f64, Vec<f64>> {
    type Output = Array<f64, Vec<f64>>;
    type View<'b>
        = ArrayView<'b, f64>
    where
        Self: 'b;

    /// Returns a reference to the element at the given multi‑dimensional indices.
    ///
    /// See [`ArrayView::get`] for full documentation on indexing behavior,
    /// error conditions, and examples. The only difference is that this method
    /// accesses the owned storage via `self.storage.as_slice()` instead of a
    /// borrowed slice.
    fn get(&self, indices: &[impl ToIndex]) -> Result<&Self::Item, Self::Error> {
        self.physical_from_indices(indices)
            .map_or(Err(ArrayError::IndexOutOfBounds), |index| {
                Ok(&self.storage.as_slice()[index])
            })
    }

    /// Slices an `ArrayView<f64>` by indices.
    ///
    /// See [`ArrayView::slice_by_indices`] for full documentation on indexing behavior,
    /// error conditions, and examples. The only difference is that this method
    /// accesses the owned storage via `self.storage.as_slice()` instead of a
    /// borrowed slice.
    fn slice_by_indices(&self, indices: &[impl ToIndex]) -> Result<Self::View<'_>, Self::Error> {
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
            data: self.storage.as_slice(),
            shape: vec![],
            strides: vec![],
            offset: new_offset,
            order: self.order,
        })
    }

    /// Slices an `ArrayView<f64>` by a range.
    ///
    /// See [`ArrayView::slice_by_range`] for full documentation on indexing behavior,
    /// error conditions, and examples. The only difference is that this method
    /// accesses the owned storage via `self.storage.as_slice()` instead of a
    /// borrowed slice.
    fn slice_by_range(
        &self,
        axis: usize,
        range: std::ops::Range<usize>,
    ) -> Result<Self::View<'_>, Self::Error> {
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
            data: self.storage.as_slice(),
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
            order: self.order,
        })
    }

    /// Slices an `ArrayView<f64>` by a stride.
    ///
    /// See [`ArrayView::slice_by_stride`] for full documentation on indexing behavior,
    /// error conditions, and examples. The only difference is that this method
    /// accesses the owned storage via `self.storage.as_slice()` instead of a
    /// borrowed slice.
    fn slice_by_stride(
        &self,
        axis: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Self::View<'_>, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }
        if step == 0 {
            return Err(ArrayError::InvalidSlice);
        }
        if start >= end || end > self.shape[axis] {
            return Err(ArrayError::InvalidSlice);
        }

        let len = (end - start).div_ceil(step);
        let mut new_shape = self.shape.clone();
        new_shape[axis] = len;
        let mut new_strides = self.strides.clone();
        new_strides[axis] = self.strides[axis] * step;
        let new_offset = self.offset + start * self.strides[axis];

        Ok(ArrayView {
            data: self.storage.as_slice(),
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
            order: self.order,
        })
    }

    ///
    /// See [`ArrayView::select`] for details.
    fn select(&self, mask: &ArrayView<'_, f64>) -> Result<Self::Output, Self::Error> {
        self.view().select(mask)
    }

    ///
    /// See [`ArrayView::take`] for details.
    fn take(&self, indices: &ArrayView<'_, f64>) -> Result<Self::Output, Self::Error> {
        self.view().take(indices)
    }

    ///
    /// See [`ArrayView::gather`] for details.
    fn gather(
        &self,
        dim: impl ToIndex,
        indices: &ArrayView<'_, f64>,
    ) -> Result<Self::Output, Self::Error> {
        self.view().gather(dim, indices)
    }

    /// Returns an iterator over the elements of the array.
    ///
    /// See [`ArrayView::iter`] for details on iteration order and performance.
    /// The only difference is that this method reads from `self.storage.as_slice()`.
    fn iter(&self) -> impl Iterator<Item = &Self::Item> {
        let pairs: Vec<(usize, usize)> =
            traversal_iters(self.shape.to_vec(), self.strides.to_vec(), self.order);
        (0..self.size()).map(move |flat: usize| {
            let mut index: usize = self.offset();
            let mut temp: usize = flat;
            for &(dim, stride) in &pairs {
                index += (temp % dim) * stride;
                temp /= dim;
            }
            &self.storage.as_slice()[index]
        })
    }

    /// Returns a reference to the element at the given multi‑dimensional indices without bounds checking.
    ///
    /// See [`ArrayView::get_unchecked`] for safety requirements and usage.
    /// This version uses the owned storage slice.
    unsafe fn get_unchecked<I: ToIndex>(&self, indices: &[I]) -> &Self::Item
    where
        I::Error: Debug,
    {
        let mut index: usize = self.offset();
        for (i, idx) in indices.iter().enumerate() {
            let dim = idx.to_index().unwrap();
            index += dim * self.strides[i];
        }
        unsafe { &*self.storage.as_slice().as_ptr().add(index) }
    }

    /// Returns a reference to the first element of the array.
    ///
    /// See [`ArrayView::first`] for behavior and examples.
    /// Differs only in accessing `self.storage.as_slice()`.
    fn first(&self) -> Result<&Self::Item, Self::Error> {
        if self.length() == 0 {
            return Err(ArrayError::EmptyArray);
        }

        Ok(&self.storage.as_slice()[self.offset])
    }

    /// Returns a reference to the last element of the array.
    ///
    /// See [`ArrayView::last`] for details on how the last logical element is computed.
    /// The underlying storage is accessed via `self.storage.as_slice()`.
    fn last(&self) -> Result<&Self::Item, Self::Error> {
        if self.length() == 0 {
            return Err(ArrayError::EmptyArray);
        }
        let mut index = self.offset;
        for d in 0..self.ndim() {
            index += (self.shape[d] - 1) * self.strides[d];
        }
        Ok(&self.storage.as_slice()[index])
    }

    /// Returns a reference to the element at the given flat logical index.
    ///
    /// See [`ArrayView::get_flat`] for explanations of flat indexing and error conditions.
    /// Uses `self.storage.as_slice()` instead of a borrowed slice.
    fn get_flat(&self, index: impl ToIndex) -> Result<&Self::Item, Self::Error> {
        let idx = index
            .to_index()
            .map_err(|_| ArrayError::IndexConversionError)?;
        let phys_idx = self.physical_from_logical_flat(idx)?;
        Ok(&self.storage.as_slice()[phys_idx])
    }

    /// Returns the underlying storage slice.
    ///
    /// For [`Array`], this returns the owned storage (e.g., from `Vec<T>`).
    /// For a view, see [`ArrayView::data`].
    fn data(&self) -> &[Self::Item] {
        self.storage.as_slice()
    }
}
