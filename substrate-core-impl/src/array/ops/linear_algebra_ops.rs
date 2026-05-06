// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use cfg_if::cfg_if;
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

    /// Computes the dot product (inner product) of two 1‑D arrays.
    ///
    /// Both arrays must be one‑dimensional and have the same length.
    /// The result is a 0‑dimensional (scalar) array containing the sum of element‑wise products.
    ///
    /// # Arguments
    /// * `other` – The right‑hand side array (must be 1‑D).
    ///
    /// # Returns
    /// `Ok(Self::Output)` containing the scalar result.
    ///
    /// # Errors
    /// * `ArrayError::ValidForVectorsOnly` – if either array is not 1‑D.
    /// * `ArrayError::IncompatibleShapes` – if the lengths differ.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, LinearAlgebraOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
    /// let dot = a.view().dot(&b.view()).unwrap();
    /// assert_eq!(dot.to_scalar().unwrap(), 32.0);
    /// ```
    fn dot(&self, other: &ArrayView<'_, f64>) -> Result<Self::Output, Self::Error> {
        if self.shape().len() != 1 || other.shape().len() != 1 {
            return Err(ArrayError::ValidForVectorsOnly);
        }

        if self.shape()[0] != other.shape()[0] {
            return Err(ArrayError::IncompatibleShapes);
        }

        cfg_if! {
            if #[cfg(feature = "simd")] {
                self.dot_simd(other)
            } else {
                self.dot_scalar(other)
            }
        }
    }

    /// Computes the matrix multiplication (matmul) of `self` and `other` (self @ other).
    /// Internally, `self` is NOT normalized to RowMajor and `other` NOT to ColumnMajor,
    /// which would maximize cache efficiency during computation. THIS CONVERSION IS UP
    /// TO THE USER. The output memory order follows RowMajor.
    ///
    /// # Arguments
    /// * `other` - A 2D array where `other.shape()[0]` must equal `self.shape()[1]`
    ///
    /// # Returns
    /// `Result<Array<E>, &'static str>` with shape `[self.shape()[0], other.shape()[1]]`
    ///
    /// # Errors
    /// - Returns `ArrayErrors::ValidForMatricesOnly` if either array is not 2D
    /// - Returns `ArrayErrors::IncompatibleShapes` if `self.shape()[1] != other.shape()[0]`
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ArrayLike;
    /// use substrate_core_spec::array::ops::{InitOps, LinearAlgebraOps, ConvertOps, AccessOps, ShapeOps};
    ///
    /// let a = Array::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap().to_row_major().unwrap();
    /// assert_eq!(a.shape(), [2, 3]);
    /// let b = Array::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap().to_column_major().unwrap();
    /// assert_eq!(b.shape(), [3, 2]);
    /// let ab = a.view().matmul(&b.view()).unwrap();
    /// assert_eq!(ab.shape(), [2, 2]);
    /// assert_eq!(ab.to_vec(), [22.0, 28.0, 49.0, 64.0]);
    /// ```
    fn matmul(&self, other: &ArrayView<'_, f64>) -> Result<Self::Output, Self::Error> {
        if self.shape().len() != 2 || other.shape().len() != 2 {
            return Err(ArrayError::ValidForMatricesOnly);
        }

        if self.shape()[1] != other.shape()[0] {
            return Err(ArrayError::IncompatibleShapes);
        }

        cfg_if! {
            if #[cfg(feature = "gpu")] {
                todo!()
            } else if #[cfg(all(feature = "parallel", feature = "simd"))] {
                return self.matmul_parallel_simd(other);
            } else if #[cfg(feature = "simd")] {
                return self.matmul_simd(other);
            } else if #[cfg(feature = "parallel")] {
                return self.matmul_parallel(other);
            } else {
                return self.matmul_scalar(other);
            }
        }
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
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
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
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]).unwrap();
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

    fn cross<Rhs: AccessOps>(&self, _other: &Rhs) -> Result<Self::Output, Self::Error> {
        todo!()
    }
}

impl LinearAlgebraOps for Array<f64, Vec<f64>> {
    type Output = Array<f64, Vec<f64>>;
    type View<'b>
        = ArrayView<'b, f64>
    where
        Self: 'b;

    /// Computes the dot product (inner product) of two 1‑D arrays.
    ///
    /// See [`ArrayView::dot`] for details.
    fn dot(&self, other: &ArrayView<'_, f64>) -> Result<Self::Output, Self::Error> {
        self.view().dot(other)
    }

    /// Computes the matrix multiplication (matmul) of `self` and `other` (self @ other).
    ///
    /// See [`ArrayView::matmul`] for details.
    fn matmul(&self, other: &ArrayView<'_, f64>) -> Result<Self::Output, Self::Error> {
        self.view().matmul(other)
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

    fn cross<Rhs: AccessOps>(&self, _other: &Rhs) -> Result<Self::Output, Self::Error> {
        todo!()
    }
}
