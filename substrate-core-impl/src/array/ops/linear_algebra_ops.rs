// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{
    ArrayLike,
    memory_order::MemoryOrder,
    ops::{AccessOps, ConvertOps, LinearAlgebraOps},
};

use crate::{
    Array,
    array::{ArrayView, error::ArrayError, utils::compute_strides},
};

impl<'a> LinearAlgebraOps for ArrayView<'a, f64> {
    type Output = Array<f64, Vec<f64>>;
    type View<'b>
        = ArrayView<'b, f64>
    where
        Self: 'b;

    fn dot<Rhs: AccessOps>(&self, other: &Rhs) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn matmul<Rhs: AccessOps>(&self, other: &Rhs) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    /// Returns a lazy transposed view of the 2D array.
    ///
    /// No data is copied; the view simply swaps the shape and strides.
    /// This operation is O(1) and zero‑cost.
    ///
    /// # Returns
    /// `Ok(ArrayView<f64>)` on success, or `Err(ArrayError::ValidForMatricesOnly)` if
    /// the array is not two‑dimensional.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, AccessOps, LinearAlgebraOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape_copy(&[2, 2]).unwrap();
    /// let t = a.transpose().unwrap();
    /// assert!(a.is_contiguous());
    /// assert!(!t.is_contiguous());
    /// assert_eq!(t.shape(), &[2, 2]);
    /// assert_eq!(t.get(&[0, 1]).unwrap(), &3.0); // original was at [1, 0]
    /// ```
    fn transpose(&self) -> Result<Self::View<'_>, Self::Error> {
        if self.ndim() != 2 {
            return Err(ArrayError::ValidForMatricesOnly);
        }

        let new_shape = vec![self.shape[1], self.shape[0]];
        let new_strides = vec![self.strides[1], self.strides[0]];

        Ok(ArrayView {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            order: self.order,
        })
    }

    /// Creates a new transposed array by copying data.
    ///
    /// This method allocates a new buffer and copies elements so that the
    /// resulting array has shape `[cols, rows]` and is **always contiguous in
    /// row‑major (C) order**, regardless of the original array's memory layout.
    /// The original array remains unchanged.
    ///
    /// # Returns
    /// `Ok(Array<f64, Vec<f64>>)` on success, or `Err(ArrayError::ValidForMatricesOnly)`
    /// if the array is not two‑dimensional.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, AccessOps, LinearAlgebraOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape_copy(&[2, 2]).unwrap();
    /// let t = a.transpose_copy().unwrap();
    /// assert_eq!(t.to_vec(), vec![1.0, 3.0, 2.0, 4.0]); // row‑major order
    /// ```
    fn transpose_copy(&self) -> Result<Self::Output, Self::Error> {
        if self.ndim() != 2 {
            return Err(ArrayError::ValidForMatricesOnly);
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let new_shape = vec![cols, rows];
        let new_size = rows * cols;
        let mut new_data = vec![0.0; new_size];

        for i in 0..rows {
            for j in 0..cols {
                let src_idx = self.offset() + i * self.strides[0] + j * self.strides[1];
                let dst_idx = j * rows + i;
                new_data[dst_idx] = self.data[src_idx];
            }
        }

        let new_strides = compute_strides(&new_shape, MemoryOrder::RowMajor);

        Ok(Array {
            storage: new_data,
            shape: new_shape,
            strides: new_strides,
            offset: 0,
            order: self.order(),
        })
    }

    fn trace(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn det(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn inv(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn solve(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn eig(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn svd(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn qr(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn cholesky(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn norm(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn cross<Rhs: AccessOps>(&self, other: &Rhs) -> Result<Self::Output, Self::Error> {
        todo!()
    }
}

impl LinearAlgebraOps for Array<f64, Vec<f64>> {
    type Output = Array<f64, Vec<f64>>;
    type View<'b>
        = ArrayView<'b, f64>
    where
        Self: 'b;

    fn dot<Rhs: AccessOps>(&self, other: &Rhs) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn matmul<Rhs: AccessOps>(&self, other: &Rhs) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    /// Returns a lazy transposed view of the 2D array.
    ///
    /// See [`ArrayView::transpose`] for details.
    fn transpose(&self) -> Result<Self::View<'_>, Self::Error> {
        if self.ndim() != 2 {
            return Err(ArrayError::ValidForMatricesOnly);
        }

        let new_shape = vec![self.shape[1], self.shape[0]];
        let new_strides = vec![self.strides[1], self.strides[0]];

        Ok(ArrayView {
            data: self.storage.as_slice(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset(),
            order: self.order(),
        })
    }

    /// Creates a new transposed array by copying data.
    ///
    /// See [`ArrayView::transpose_copy`] for details.
    fn transpose_copy(&self) -> Result<Self::Output, Self::Error> {
        self.view().transpose_copy()
    }

    fn trace(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn det(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn inv(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn solve(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn eig(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn svd(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn qr(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn cholesky(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn norm(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    fn cross<Rhs: AccessOps>(&self, other: &Rhs) -> Result<Self::Output, Self::Error> {
        todo!()
    }
}
