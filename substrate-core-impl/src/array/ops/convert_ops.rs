// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{ArrayLike, ops::AccessOps, ops::ConvertOps};

use crate::{
    Array,
    array::{ArrayView, error::ArrayError},
};

impl<'a> ConvertOps for ArrayView<'a, f64> {
    type View<'b>
        = ArrayView<'b, f64>
    where
        Self: 'b;

    /// Converts the array to a scalar value.
    ///
    /// This method succeeds only if the array contains exactly one element
    /// (i.e., `length() == 1`). It returns that element by value.
    ///
    /// # Returns
    /// `Ok(Self::Item)` if the array is a scalar (single element),
    /// otherwise `Err(ArrayError::ArrayNotAScalar)`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![42.0]);
    /// let a_view = a.view();
    /// assert_eq!(a_view.to_scalar().unwrap(), 42.0);
    ///
    /// let b = Array::from_vec(vec![1.0, 2.0]);
    /// let b_view = b.view();
    /// assert!(b_view.to_scalar().is_err());
    /// ```
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

    /// Copies the entire underlying storage into a new `Vec<Self::Item>`.
    ///
    /// This method allocates a new vector and clones each element from the
    /// owned storage slice. The resulting vector is independent of the array
    /// and can be used for further processing or serialisation.
    ///
    /// # Returns
    /// A `Vec<Self::Item>` containing a copy of all elements in the storage.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let a_view = a.view();
    /// let mut v = a.to_vec();
    /// assert_eq!(v, vec![1.0, 2.0, 3.0]);
    ///
    /// // Modifying the copy does not affect the original array
    /// v[0] = 99.0;
    /// assert_eq!(a_view.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    fn to_vec(&self) -> Vec<Self::Item> {
        self.data.to_vec()
    }

    /// Create an `ArrayView` from an `ArrayView`.
    fn view(&self) -> Self::View<'_> {
        self.clone()
    }
}

impl ConvertOps for Array<f64, Vec<f64>> {
    type View<'a> = ArrayView<'a, f64>;

    /// Converts the array to a scalar value.
    ///
    /// See [`ArrayView::to_scalar`] for details.
    fn to_scalar(&self) -> Result<Self::Item, Self::Error> {
        self.view().to_scalar()
    }

    /// Copies the entire underlying storage into a new `Vec<Self::Item>`.
    ///
    /// See [`ArrayView::to_vec`] for details.
    fn to_vec(&self) -> Vec<Self::Item> {
        self.view().to_vec()
    }

    /// Create an `ArrayView` from an `Array`.
    ///
    /// # Returns
    /// `ArrayView<f64>`
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ArrayLike;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps, AccessOps};
    ///
    /// let a = Array::zeros(&[2, 2]);
    /// let a_view = a.view();
    /// assert_eq!(a_view.shape(), [2, 2]);
    /// assert_eq!(a_view.strides(), [2, 1]);
    /// assert_eq!(a_view.data(), [0.0, 0.0, 0.0, 0.0]);
    /// assert_eq!(a_view.ndim(), 2);
    /// assert_eq!(a_view.size(), 4);
    /// ```
    fn view(&self) -> ArrayView<'_, f64> {
        ArrayView {
            data: self.storage.as_slice(),
            shape: self.shape.to_vec(),
            strides: self.strides.to_vec(),
            offset: self.offset,
            order: self.order,
        }
    }
}
