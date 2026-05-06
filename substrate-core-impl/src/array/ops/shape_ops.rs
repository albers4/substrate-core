// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{
    ArrayLike, ArrayViewLike,
    memory_order::MemoryOrder,
    ops::{AccessOps, ConvertOps, ShapeOps},
    pad_mode::PadMode,
};

use crate::{
    Array,
    array::{
        ArrayView,
        error::ArrayError,
        utils::{broadcast_strides, compute_strides, unravel_index},
    },
};

impl<'a> ShapeOps for ArrayView<'a, f64> {
    type Output = Array<f64, Vec<f64>>;
    type View<'b>
        = ArrayView<'b, f64>
    where
        Self: 'b;

    /// Reshapes the array view into a new shape without copying data.
    ///
    /// This method returns a new **view** that reinterprets the same underlying data
    /// with the given shape. No memory allocation or data movement occurs.
    ///
    /// # Prerequisites
    /// - The array must be **contiguous** in memory (i.e., `is_contiguous()` returns `true`).
    /// - The total number of elements (`length()`) must equal the product of `new_shape`.
    /// - No dimension in `new_shape` may be zero.
    /// - `new_shape` cannot be empty (use a scalar view for 0‑dimensional arrays instead).
    ///
    /// # Errors
    /// - `ArrayError::EmptyShape` – if `new_shape` is empty.
    /// - `ArrayError::InvalidShapeDimension` – if any dimension is zero.
    /// - `ArrayError::ReshapeSizeMismatch` – if the product of `new_shape` does not match `self.length()`.
    /// - `ArrayError::NotContiguous` – if the array is not contiguous.
    ///
    /// # Returns
    /// A new `ArrayView<f64>` (the associated `View` type) with the specified shape and
    /// the same memory order (`RowMajor` or `ColumnMajor`) as the original.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps, AccessOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let view = a.view();
    /// let reshaped = view.reshape(&[2, 3]).unwrap();
    /// assert_eq!(reshaped.shape(), &[2, 3]);
    /// assert_eq!(reshaped.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    fn reshape_view(&self, new_shape: &[usize]) -> Result<Self::View<'_>, Self::Error> {
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
        Ok(ArrayView {
            data: self.data,
            shape: new_shape.to_vec(),
            strides,
            offset: self.offset(),
            order: self.order(),
        })
    }

    /// Reshapes the array view into a new shape, returning an owned array.
    ///
    /// This method copies the underlying data into a new buffer and reinterprets it
    /// with the given shape. The view **must be contiguous** (no gaps between elements);
    /// otherwise an error is returned. The resulting array is always contiguous.
    ///
    /// # Arguments
    /// * `new_shape` - The desired shape. The total number of elements must match `self.length()`.
    ///
    /// # Errors
    /// * `ArrayError::InvalidShapeDimension` – if any dimension is zero.
    /// * `ArrayError::ReshapeSizeMismatch` – if the product of `new_shape` differs from `self.length()`.
    /// * `ArrayError::NotContiguous` – if the view is not contiguous.
    /// * `ArrayError::EmptyShape` – if `new_shape` is empty.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the specified shape, row‑major strides (if
    /// the original memory order can be preserved), and a copy of the data.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let view = a.view();
    /// let reshaped = view.reshape(&[2, 3]).unwrap();
    /// assert_eq!(reshaped.shape(), &[2, 3]);
    /// assert_eq!(reshaped.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    fn reshape(self, new_shape: &[usize]) -> Result<Self::Output, Self::Error> {
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
            storage: self.data.to_vec(),
            shape: new_shape.to_vec(),
            strides,
            offset: self.offset(),
            order: self.order(),
        })
    }

    /// Converts the array view into a contiguous row‑major owned array.
    ///
    /// If the view is already in canonical row‑major order (contiguous with row‑major strides),
    /// this method returns a copy via `into_owned()`. Otherwise, it allocates a new buffer
    /// and copies the elements in row‑major logical order.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` that is guaranteed to be contiguous and row‑major.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps, ShapeOps, LinearAlgebraOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// let view = a.view();
    /// let transposed_view = view.transpose().unwrap();         // lazy, non‑contiguous
    /// assert!(!transposed_view.is_contiguous());
    /// let row_major = transposed_view.to_row_major().unwrap(); // now contiguous row‑major
    /// assert!(row_major.is_contiguous());
    /// assert_eq!(row_major.shape(), &[2, 2]);
    /// assert_eq!(row_major.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    /// ```
    fn to_row_major(self) -> Result<Self::Output, Self::Error> {
        if self.is_canonical(MemoryOrder::RowMajor) {
            return Ok(self.into_owned());
        }

        let mut row_major_storage = vec![0.0f64; self.length()];

        for (i, dst) in row_major_storage.iter_mut().enumerate() {
            let row_indices = unravel_index(i, self.shape(), MemoryOrder::RowMajor)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            let src_index: usize = self
                .physical_from_indices(&row_indices)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            *dst = self.data[src_index];
        }

        Ok(Array {
            storage: row_major_storage,
            shape: self.shape.to_vec(),
            strides: compute_strides(self.shape(), MemoryOrder::RowMajor),
            offset: 0,
            order: MemoryOrder::RowMajor,
        })
    }

    ///
    /// This does not consume the view and is safe to call from a shared reference.
    /// If the view is already in canonical row‑major order (contiguous, correct strides),
    /// this method copies the data directly via the iterator (still copies, but avoids the
    /// expensive per‑index mapping). Otherwise, it uses the general indexing loop.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` in row‑major (C) order, always contiguous.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps, ShapeOps};
    /// use substrate_core_spec::array::ArrayLike;
    /// use substrate_core_spec::array::memory_order::MemoryOrder;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .reshape(&[2, 3])
    ///     .unwrap();
    /// let a_row = a.view().to_row_major_copy().unwrap();
    /// assert_eq!(a_row.shape(), &[2, 3]);
    /// assert_eq!(a_row.order(), MemoryOrder::RowMajor);
    /// assert_eq!(a_row.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    fn to_row_major_copy(&self) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        if self.is_canonical(MemoryOrder::RowMajor) {
            // Fast path: already row‑major contiguous. Copy directly via iterator.
            let data = self.iter().copied().collect();
            return Array::from_vec_with_shape(data, self.shape());
        }

        let mut row_major_storage = vec![0.0; self.length()];
        for (i, dst) in row_major_storage.iter_mut().enumerate() {
            let row_indices = unravel_index(i, self.shape(), MemoryOrder::RowMajor)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            let src_index = self
                .physical_from_indices(&row_indices)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            *dst = self.data[src_index];
        }

        Ok(Array {
            storage: row_major_storage,
            shape: self.shape().to_vec(),
            strides: compute_strides(self.shape(), MemoryOrder::RowMajor),
            offset: 0,
            order: MemoryOrder::RowMajor,
        })
    }

    /// Converts the array view into a contiguous column‑major owned array.
    ///
    /// If the view is already in canonical column‑major order (contiguous with column‑major strides),
    /// this method returns a copy via `into_owned()`. Otherwise, it allocates a new buffer
    /// and copies the elements in column‑major logical order.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` that is guaranteed to be contiguous and column‑major.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps, ShapeOps, LinearAlgebraOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap(); // this gives row-major
    /// let view = a.view();
    /// let transposed_view = view.transpose().unwrap();       // lazy, non‑contiguous
    /// assert!(!transposed_view.is_contiguous());
    /// let col_major = transposed_view.to_column_major().unwrap();
    /// assert!(col_major.is_contiguous());
    /// assert_eq!(col_major.shape(), &[2, 2]);
    /// assert_eq!(col_major.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    fn to_column_major(self) -> Result<Self::Output, Self::Error> {
        if self.is_canonical(MemoryOrder::ColumnMajor) {
            return Ok(self.into_owned());
        }

        let mut column_major_storage = vec![0.0f64; self.length()];

        for (i, dst) in column_major_storage.iter_mut().enumerate() {
            let column_indices = unravel_index(i, self.shape(), MemoryOrder::ColumnMajor)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            let src_index: usize = self
                .physical_from_indices(&column_indices)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            *dst = self.data[src_index];
        }

        Ok(Array {
            storage: column_major_storage,
            shape: self.shape.to_vec(),
            strides: compute_strides(self.shape(), MemoryOrder::ColumnMajor),
            offset: 0,
            order: MemoryOrder::ColumnMajor,
        })
    }

    /// Copies the view into a contiguous column‑major owned array.
    ///
    /// This does not consume the view and is safe to call from a shared reference.
    /// If the view is already in canonical column‑major order (contiguous, correct strides),
    /// this method copies the data directly via the iterator (still copies, but avoids the
    /// expensive per‑index mapping). Otherwise, it uses the general indexing loop.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` in column‑major (Fortran) order, always contiguous.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps, ShapeOps};
    /// use substrate_core_spec::array::ArrayLike;
    /// use substrate_core_spec::array::memory_order::MemoryOrder;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .reshape(&[2, 3])
    ///     .unwrap();
    /// let a_col = a.view().to_column_major_copy().unwrap();
    /// assert_eq!(a_col.shape(), &[2, 3]);
    /// assert_eq!(a_col.order(), MemoryOrder::ColumnMajor);
    /// // Column‑major order: column by column -> [1,4,2,5,3,6]
    /// assert_eq!(a_col.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    fn to_column_major_copy(&self) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        if self.is_canonical(MemoryOrder::ColumnMajor) {
            // Fast path: already column‑major contiguous. Use iterator copy.
            let data = self.iter().copied().collect();
            return Array::from_vec_with_shape(data, self.shape());
        }
        let mut col_major_storage = vec![0.0; self.length()];
        for (i, dst) in col_major_storage.iter_mut().enumerate() {
            let col_indices = unravel_index(i, self.shape(), MemoryOrder::ColumnMajor)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            let src_index = self
                .physical_from_indices(&col_indices)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            *dst = self.data[src_index];
        }
        Ok(Array {
            storage: col_major_storage,
            shape: self.shape().to_vec(),
            strides: compute_strides(self.shape(), MemoryOrder::ColumnMajor),
            offset: 0,
            order: MemoryOrder::ColumnMajor,
        })
    }

    /// Flattens the array into a 1‑dimensional array (row‑major order).
    ///
    /// The new array contains all elements of the original view in logical order,
    /// preserving the sequence defined by the view’s memory layout (row‑major or column‑major).
    /// The resulting shape is `[len]`, where `len = self.length()`.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with shape `[self.length()]`.
    ///
    /// # Errors
    /// This method cannot fail under normal circumstances, but may propagate errors
    /// from internal iterators (unlikely). Returns `ArrayError` if shape construction fails.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
    /// let flat = a.view().flatten().unwrap();
    /// assert_eq!(flat.shape(), &[4]);
    /// assert_eq!(flat.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    fn flatten(&self) -> Result<Self::Output, Self::Error> {
        let data = self.iter().copied().collect::<Vec<f64>>();
        let shape = vec![data.len()];

        Array::from_vec_with_shape(data, &shape)
    }

    /// Removes dimensions of size 1 from the array shape.
    ///
    /// All axes where `shape[i] == 1` are removed. If the resulting shape is empty
    /// (i.e., the original array was a scalar or became a scalar), the output is a
    /// 0‑dimensional array (scalar) containing the single element.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with all singleton dimensions squeezed out.
    ///
    /// # Errors
    /// Returns `ArrayError::EmptyArray` if the original view is empty (should not happen
    /// for a valid view). May also propagate errors from internal operations.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3, 1]).unwrap();
    /// let squeezed = a.view().squeeze().unwrap();
    /// assert_eq!(squeezed.shape(), &[3]);
    /// assert_eq!(squeezed.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    fn squeeze(&self) -> Result<Self::Output, Self::Error> {
        let new_shape: Vec<usize> = self.shape().iter().filter(|&&d| d != 1).copied().collect();
        if new_shape.is_empty() {
            let scalar = *self.iter().next().ok_or(ArrayError::EmptyArray)?;
            return Ok(Array::from_scalar(scalar));
        }
        let new_len = new_shape.iter().product::<usize>();
        let mut data = Vec::with_capacity(new_len);

        for flat_idx in 0..self.length() {
            let val = *self.get_flat(flat_idx)?;
            data.push(val);
        }

        let strides = compute_strides(&new_shape, self.order());
        Ok(Array {
            storage: data,
            shape: new_shape,
            strides,
            offset: 0,
            order: self.order(),
        })
    }

    /// Adds a new dimension of size 1 at the specified axis position.
    ///
    /// The new axis is inserted before the existing axis with the given index.
    /// For `axis == ndim()`, the new dimension is appended at the end.
    ///
    /// # Arguments
    /// * `axis` – Position where the new dimension is inserted (0 ≤ axis ≤ ndim).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with one extra dimension of length 1.
    ///
    /// # Errors
    /// Returns `ArrayError::AxisOutOfBounds` if `axis > ndim()`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let unsqueezed = a.view().unsqueeze(0).unwrap();
    /// assert_eq!(unsqueezed.shape(), &[1, 3]);
    /// assert_eq!(unsqueezed.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    fn unsqueeze(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        let ndim = self.ndim();
        if axis > ndim {
            return Err(ArrayError::AxisOutOfBounds);
        }

        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, 1);
        let new_len = new_shape.iter().product::<usize>();
        let mut data = Vec::with_capacity(new_len);

        for flat_idx in 0..self.length() {
            let val = *self.get_flat(flat_idx)?;
            data.push(val);
        }

        Array::from_vec_with_shape(data, &new_shape)
    }

    /// Broadcasts the view to a target shape, copying data where necessary.
    ///
    /// The view is treated as a pattern that is repeated along axes where its shape
    /// is `1` and the target shape is larger. Axes of length 1 in the view can be
    /// expanded to any size; all other dimensions must match exactly. The result is
    /// a new owned array with the given shape, filled by repeating the view’s elements.
    ///
    /// # Arguments
    /// * `shape` – The target shape. Must be broadcast‑compatible with the view’s shape.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` of shape `shape` containing the broadcasted data.
    ///
    /// # Errors
    /// * `ArrayError::IncompatibleShapes` – if the shapes are not broadcast‑compatible.
    /// * `ArrayError::DimensionMismatch` – if the number of dimensions differs in an incompatible way.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// // Shape [3] can broadcast to [2, 3] (repeat once along axis 0)
    /// let b = a.view().broadcast_to(&[2, 3]).unwrap();
    /// assert_eq!(b.shape(), &[2, 3]);
    /// assert_eq!(b.to_vec(), vec![1.0,2.0,3.0, 1.0,2.0,3.0]);
    /// ```
    fn broadcast_to(&self, shape: &[usize]) -> Result<Self::Output, Self::Error> {
        let target_shape = shape.to_vec();
        let total_len = target_shape.iter().product::<usize>();
        let strides = broadcast_strides(&self.shape, &self.strides, &target_shape)?;
        let offset = self.offset();
        let mut data = vec![0.0; total_len];

        for (flat_idx, flat_item) in data.iter_mut().enumerate() {
            let mut rem = flat_idx;
            let mut idx = offset;
            for dim in (0..target_shape.len()).rev() {
                let coord = rem % target_shape[dim];
                rem /= target_shape[dim];
                idx += coord * strides[dim];
            }
            *flat_item = unsafe { *self.data.as_ptr().add(idx) };
        }

        Array::from_vec_with_shape(data, &target_shape)
    }

    /// Concatenates the view with a list of other views along the specified axis.
    ///
    /// All arrays must have the same number of dimensions, and all dimensions except
    /// the concatenation axis must be equal. The output shape is the same as the input
    /// shape, except that the size along `axis` becomes the sum of the sizes of all
    /// concatenated arrays.
    ///
    /// # Arguments
    /// * `arrays` – A slice of views to concatenate after `self` (order is preserved).
    /// * `axis` – The axis along which to concatenate (0‑based).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` containing the concatenated data.
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `axis >= ndim()`.
    /// * `ArrayError::IncompatibleShapes` – if any array’s shape differs in dimensions other than `axis`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0]);
    /// let b = Array::from_vec(vec![3.0, 4.0]);
    /// let c = a.view().concatenate(&[b.view()], 0).unwrap();
    /// assert_eq!(c.shape(), &[4]);
    /// assert_eq!(c.to_vec(), vec![1.0,2.0,3.0,4.0]);
    /// ```
    fn concatenate(
        &self,
        arrays: &[ArrayView<'_, f64>],
        axis: usize,
    ) -> Result<Self::Output, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }

        let mut total_size = self.shape()[axis];
        for arr in arrays {
            if arr.ndim() != self.ndim() {
                return Err(ArrayError::IncompatibleShapes);
            }
            for d in 0..self.ndim() {
                if d != axis && arr.shape()[d] != self.shape()[d] {
                    return Err(ArrayError::IncompatibleShapes);
                }
            }
            total_size += arr.shape()[axis];
        }

        let mut new_shape = self.shape().to_vec();
        new_shape[axis] = total_size;
        let total_len = new_shape.iter().product::<usize>();
        let mut data = vec![0.0; total_len];
        let mut out_flat = 0;

        for arr in std::iter::once(self).chain(arrays) {
            let arr_len = arr.length();
            for i in 0..arr_len {
                data[out_flat] = *arr.get_flat(i)?;
                out_flat += 1;
            }
        }

        Array::from_vec_with_shape(data, &new_shape)
    }

    /// Stacks the view and a list of other views along a new axis.
    ///
    /// All arrays must have identical shapes. A new axis of length `(arrays.len() + 1)`
    /// is inserted at the specified position. The array `self` becomes the first element
    /// along the new axis, followed by the arrays in the given order.
    ///
    /// # Arguments
    /// * `arrays` – A slice of views to stack after `self`.
    /// * `axis` – Position where the new axis is inserted (0 ≤ axis ≤ ndim).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` whose rank is one higher than the original.
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `axis > ndim()`.
    /// * `ArrayError::IncompatibleShapes` – if any array’s shape differs from `self.shape()`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0]);
    /// let b = Array::from_vec(vec![3.0, 4.0]);
    /// let stacked = a.view().stack(&[b.view()], 0).unwrap();
    /// assert_eq!(stacked.shape(), &[2, 2]);
    /// assert_eq!(stacked.to_vec(), vec![1.0,2.0,3.0,4.0]);
    /// ```
    fn stack(
        &self,
        arrays: &[ArrayView<'_, f64>],
        axis: usize,
    ) -> Result<Self::Output, Self::Error> {
        if axis > self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }

        for arr in arrays {
            if arr.shape() != self.shape() {
                return Err(ArrayError::IncompatibleShapes);
            }
        }

        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, arrays.len() + 1);
        let total_len = new_shape.iter().product::<usize>();
        let mut data = vec![0.0; total_len];
        let mut out_flat = 0;

        for arr in std::iter::once(self).chain(arrays) {
            for i in 0..arr.length() {
                data[out_flat] = *arr.get_flat(i)?;
                out_flat += 1;
            }
        }

        Array::from_vec_with_shape(data, &new_shape)
    }

    /// Splits the array into multiple sub‑arrays along the given axis.
    ///
    /// The axis is split into `indices_or_sections` equal parts (must divide evenly).
    /// The result is a vector of arrays, each having the same shape as the original,
    /// except that the dimension along `axis` is reduced to the part size.
    ///
    /// # Arguments
    /// * `indices_or_sections` – Number of equal parts to split into.
    /// * `axis` – Axis along which to split (0‑based).
    ///
    /// # Returns
    /// A `Vec<Array<f64, Vec<f64>>>` containing the sub‑arrays in order.
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `axis >= ndim()`.
    /// * `ArrayError::InvalidSplit` – if the axis size is not divisible by `indices_or_sections`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0,2.0,3.0,4.0,5.0,6.0]).reshape(&[2,3]).unwrap();
    /// let splits = a.view().split(3, 1).unwrap();
    /// assert_eq!(splits.len(), 3);
    /// assert_eq!(splits[0].to_vec(), vec![1.0, 4.0]);
    /// assert_eq!(splits[1].to_vec(), vec![2.0, 5.0]);
    /// assert_eq!(splits[2].to_vec(), vec![3.0, 6.0]);
    /// ```
    fn split(
        &self,
        indices_or_sections: usize,
        axis: usize,
    ) -> Result<Vec<Self::Output>, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }

        let dim_size = self.shape()[axis];
        if !dim_size.is_multiple_of(indices_or_sections) {
            return Err(ArrayError::InvalidSplit);
        }

        let chunk = dim_size / indices_or_sections;
        let mut result = Vec::new();

        for start in (0..dim_size).step_by(chunk) {
            let mut new_shape = self.shape().to_vec();
            new_shape[axis] = chunk;
            let mut data = Vec::with_capacity(chunk * self.length() / dim_size);

            for flat_idx in 0..self.length() {
                let idx = self.physical_from_logical_flat(flat_idx)?;
                let coords = unravel_index(flat_idx, self.shape(), self.order())?;
                if coords[axis] >= start && coords[axis] < start + chunk {
                    data.push(self.data[idx]);
                }
            }
            let arr = Array::from_vec_with_shape(data, &new_shape)?;
            result.push(arr);
        }

        Ok(result)
    }

    /// Rolls array elements along a specified axis, or along the flattened array if no axis is given.
    ///
    /// Elements that are shifted beyond the last position wrap around to the beginning.
    /// A positive shift moves elements forward (rightwards for axis=1), negative shift moves
    /// backwards. When `axis` is `None`, the array is flattened, rolled, and reshaped back.
    ///
    /// **Note:** Currently only full flat roll (`axis=None`) is implemented.
    /// Rolling along a specific axis is not yet available and will return `NotImplemented`.
    ///
    /// # Arguments
    /// * `shift` – Number of positions to shift (positive = forward, negative = backward).
    /// * `axis` – Axis along which to roll (optional). If `None`, rolls over the flattened array.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the rolled elements.
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `axis` is `Some(ax)` and `ax >= ndim()`.
    /// * `ArrayError::NotImplemented` – if `axis` is `Some` (not yet implemented).
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0,2.0,3.0,4.0]);
    /// let rolled = a.view().roll(2, None).unwrap();
    /// assert_eq!(rolled.to_vec(), vec![3.0,4.0,1.0,2.0]);
    /// ```
    fn roll(&self, shift: isize, axis: Option<usize>) -> Result<Self::Output, Self::Error> {
        let total_len = self.length();
        let mut data = vec![0.0; total_len];

        if let Some(ax) = axis {
            if ax >= self.ndim() {
                return Err(ArrayError::AxisOutOfBounds);
            }
            // 1D roll along one axis: need to handle subarrays
            // For simplicity, we only support full roll (axis=None) or 1D array.
            // For multidimensional, it's complex. Implementatin will follow later.
            return Err(ArrayError::NotImplemented);
        }
        // Full flat roll
        let shift_mod = shift.rem_euclid(total_len as isize) as usize;
        for (i, item) in data.iter_mut().enumerate() {
            let src = (i + total_len - shift_mod) % total_len;
            *item = *self.get_flat(src)?;
        }

        let shape = self.shape().to_vec();
        Array::from_vec_with_shape(data, &shape)
    }

    /// Pads the array with constant values around its edges.
    ///
    /// For each dimension, a pair `(before, after)` specifies how many elements to add
    /// before and after the original data. The padding is filled with a constant value
    /// specified by the `PadMode::Constant`. Other padding modes are not yet implemented.
    ///
    /// # Arguments
    /// * `pad_width` – A slice of `(usize, usize)` of length equal to `ndim()`, specifying
    ///   the number of padding elements before and after each axis.
    /// * `mode` – Padding mode. Currently only `PadMode::Constant(value)` is supported.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the padded shape.
    ///
    /// # Errors
    /// * `ArrayError::DimensionMismatch` – if `pad_width.len() != ndim()`.
    /// * `ArrayError::NotImplemented` – if `mode` is not `PadMode::Constant`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::pad_mode::PadMode;
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0,2.0,3.0]);
    /// let padded = a.view().pad(&[(1,1)], PadMode::Constant(0.0)).unwrap();
    /// assert_eq!(padded.shape(), &[5]);
    /// assert_eq!(padded.to_vec(), vec![0.0,1.0,2.0,3.0,0.0]);
    /// ```
    fn pad(
        &self,
        pad_width: &[(usize, usize)],
        mode: PadMode,
    ) -> Result<Self::Output, Self::Error> {
        let ndim = self.ndim();
        if pad_width.len() != ndim {
            return Err(ArrayError::DimensionMismatch);
        }

        let mut new_shape = Vec::with_capacity(ndim);
        for (d, item) in pad_width.iter().enumerate().take(ndim) {
            let (before, after) = item;
            new_shape.push(self.shape()[d] + before + after);
        }

        let total_len = new_shape.iter().product::<usize>();
        let mut data = vec![0.0; total_len];
        let out_strides = compute_strides(&new_shape, self.order());

        if let PadMode::Constant(constant) = mode {
            for item in data.iter_mut() {
                *item = constant;
            }
        } else {
            return Err(ArrayError::NotImplemented);
        }

        for flat_idx in 0..self.length() {
            let coords = unravel_index(flat_idx, self.shape(), self.order())?;
            let mut out_coords = coords.clone();

            for d in 0..ndim {
                out_coords[d] += pad_width[d].0;
            }
            let out_flat = out_coords
                .iter()
                .enumerate()
                .fold(0, |idx, (i, &c)| idx + c * out_strides[i]);
            data[out_flat] = *self.get_flat(flat_idx)?;
        }
        Array::from_vec_with_shape(data, &new_shape)
    }

    /// Repeats the array multiple times, tiling it according to the given repetition counts.
    ///
    /// The `reps` parameter specifies how many times to repeat the array along each axis.
    /// If `reps` has more dimensions than the array, new axes of length 1 are inserted
    /// at the beginning to match the length of `reps`. The resulting shape is
    /// `shape[i] * reps[i]` for each axis.
    ///
    /// # Arguments
    /// * `reps` – Number of repetitions along each axis (length at least `ndim()`).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` tiled as specified.
    ///
    /// # Errors
    /// * `ArrayError::DimensionMismatch` – if `reps.len() < ndim()`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0,2.0]);
    /// let tiled = a.view().tile(&[3]).unwrap();
    /// assert_eq!(tiled.shape(), &[6]);
    /// assert_eq!(tiled.to_vec(), vec![1.0,2.0, 1.0,2.0, 1.0,2.0]);
    /// ```
    fn tile(&self, reps: &[usize]) -> Result<Self::Output, Self::Error> {
        let ndim = self.ndim();
        if reps.len() < ndim {
            return Err(ArrayError::DimensionMismatch);
        }

        let mut new_shape = Vec::with_capacity(reps.len());
        for (d, rep) in reps.iter().enumerate() {
            if d < ndim {
                new_shape.push(self.shape()[d] * rep);
            } else {
                new_shape.push(*rep);
            }
        }

        let total_len = new_shape.iter().product::<usize>();
        let mut data = vec![0.0; total_len];
        let out_strides = compute_strides(&new_shape, self.order());

        for (out_flat, out_item) in data.iter_mut().enumerate() {
            let out_coords = unravel_index(out_flat, &new_shape, self.order())?;
            let mut in_coords = Vec::with_capacity(ndim);

            for (d, out_coord) in out_coords.iter().enumerate() {
                in_coords.push(out_coord % self.shape()[d]);
            }
            let in_flat = self.physical_from_indices(&in_coords)?;
            *out_item = self.data[in_flat];
        }

        Ok(Array {
            storage: data,
            shape: new_shape,
            strides: out_strides,
            offset: 0,
            order: self.order(),
        })
    }
}
impl ShapeOps for Array<f64, Vec<f64>> {
    type Output = Array<f64, Vec<f64>>;
    type View<'b>
        = ArrayView<'b, f64>
    where
        Self: 'b;

    /// Zero-copy reshape, return a new view.
    ///
    /// See [`ArrayView::reshape`] for details.
    fn reshape_view(&self, new_shape: &[usize]) -> Result<Self::View<'_>, Self::Error> {
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
        Ok(ArrayView {
            data: self.storage.as_slice(),
            shape: new_shape.to_vec(),
            strides,
            offset: self.offset(),
            order: self.order(),
        })
    }

    /// Reshapes the array view into a new shape, returning an owned array.
    ///
    /// See [`ArrayView::reshape`] for details.
    fn reshape(self, new_shape: &[usize]) -> Result<Self::Output, Self::Error> {
        self.view().reshape(new_shape)
    }

    /// Converts the array view into a contiguous row‑major owned array.
    ///
    /// See [`ArrayView::to_row_major`] for details.
    fn to_row_major(self) -> Result<Self::Output, Self::Error> {
        self.view().to_row_major()
    }

    /// Copies the view into a contiguous row‑major owned array.
    ///
    /// See [`ArrayView::to_row_major_copy`] for details.
    fn to_row_major_copy(&self) -> Result<Self::Output, Self::Error> {
        self.view().to_row_major_copy()
    }

    /// Converts the array view into a contiguous row‑major owned array.
    ///
    /// See [`ArrayView::to_column_major`] for details.
    fn to_column_major(self) -> Result<Self::Output, Self::Error> {
        self.view().to_column_major()
    }

    /// Copies the view into a contiguous column‑major owned array.
    ///
    /// See [`ArrayView::to_column_major`] for details.
    fn to_column_major_copy(&self) -> Result<Self::Output, Self::Error> {
        self.view().to_column_major_copy()
    }

    /// Flattens the array into a 1‑dimensional array (row‑major order).
    ///
    /// See [`ArrayView::flatten`] for details.
    fn flatten(&self) -> Result<Self::Output, Self::Error> {
        self.view().flatten()
    }

    /// Removes dimensions of size 1 from the array shape.
    ///
    /// See [`ArrayView::squeeze`] for details.
    fn squeeze(&self) -> Result<Self::Output, Self::Error> {
        self.view().squeeze()
    }

    /// Adds a new dimension of size 1 at the specified axis position.
    ///
    /// See [`ArrayView::unsqueeze`] for details.
    fn unsqueeze(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        self.view().unsqueeze(axis)
    }

    /// Broadcasts the view to a target shape, copying data where necessary.
    ///
    /// See [`ArrayView::broadcast_to`] for details.
    fn broadcast_to(&self, shape: &[usize]) -> Result<Self::Output, Self::Error> {
        self.view().broadcast_to(shape)
    }

    /// Concatenates the view with a list of other views along the specified axis.
    ///
    /// See [`ArrayView::concatenate`] for details.
    fn concatenate(
        &self,
        arrays: &[ArrayView<'_, f64>],
        axis: usize,
    ) -> Result<Self::Output, Self::Error> {
        self.view().concatenate(arrays, axis)
    }

    /// Stacks the view and a list of other views along a new axis.
    ///
    /// See [`ArrayView::stack`] for details.
    fn stack(
        &self,
        arrays: &[ArrayView<'_, f64>],
        axis: usize,
    ) -> Result<Self::Output, Self::Error> {
        self.view().stack(arrays, axis)
    }

    /// Splits the array into multiple sub‑arrays along the given axis.
    ///
    /// See [`ArrayView::split`] for details.
    fn split(
        &self,
        indices_or_sections: usize,
        axis: usize,
    ) -> Result<Vec<Self::Output>, Self::Error> {
        self.view().split(indices_or_sections, axis)
    }

    /// Rolls array elements along a specified axis, or along the flattened array if no axis is given.
    ///
    /// See [`ArrayView::roll`] for details.
    fn roll(&self, shift: isize, axis: Option<usize>) -> Result<Self::Output, Self::Error> {
        self.view().roll(shift, axis)
    }

    ///
    /// See [`ArrayView::pad`] for details.
    fn pad(
        &self,
        pad_width: &[(usize, usize)],
        mode: PadMode,
    ) -> Result<Self::Output, Self::Error> {
        self.view().pad(pad_width, mode)
    }

    ///
    /// See [`ArrayView::tile`] for details.
    fn tile(&self, reps: &[usize]) -> Result<Self::Output, Self::Error> {
        self.view().tile(reps)
    }
}
