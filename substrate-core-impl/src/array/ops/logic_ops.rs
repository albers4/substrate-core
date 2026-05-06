// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{
    ArrayLike,
    ops::{AccessOps, ConvertOps, LogicOps},
};

use crate::{
    Array,
    array::{
        ArrayView,
        error::ArrayError,
        utils::{broadcast_shapes, broadcast_strides},
    },
};

impl<'a> LogicOps for ArrayView<'a, f64> {
    /// Returns `true` if all elements of the array are finite (not infinite and not NaN).
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, LogicOps};
    ///
    /// let is_finite = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    /// let is_infinite = Array::from_vec(vec![0.0, 1.0, 2.0, f64::INFINITY]);
    /// assert!(is_finite.is_finite());
    /// assert!(!is_infinite.is_finite());
    /// ```
    fn is_finite(&self) -> bool {
        self.iter().all(|x| x.is_finite())
    }

    /// Returns `true` if all elements of the array are infinite (either positive or negative infinity).
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, LogicOps};
    ///
    /// let is_finite = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    /// let is_infinite = Array::from_vec(vec![0.0, 1.0, 2.0, f64::INFINITY]);
    /// assert!(is_finite.is_finite());
    /// assert!(!is_infinite.is_finite());
    /// ```
    fn is_inf(&self) -> bool {
        self.iter().all(|x| x.is_infinite())
    }

    /// Returns `true` if all elements of the array are NaN.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, LogicOps};
    ///
    /// let is_nan = Array::from_vec(vec![f64::NAN; 4]);
    /// let is_number = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    /// assert!(is_nan.is_nan());
    /// assert!(!is_number.is_nan());
    /// ```
    fn is_nan(&self) -> bool {
        self.iter().all(|x| x.is_nan())
    }

    /// Returns `true` if all corresponding elements are close within absolute and relative tolerances.
    ///
    /// Default tolerances: `rtol = 1e-5`, `atol = 1e-8`. For custom tolerances, use `allclose_with`.
    /// The shapes must be broadcastable. The function returns `false` if shapes are incompatible.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, LogicOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![1.00000001, 2.00000001, 3.00000001]);
    /// assert!(a.allclose(&b).unwrap());
    ///
    /// let c = Array::from_vec(vec![1.0, 2.1, 3.0]);
    /// assert!(!a.allclose(&c).unwrap());
    /// ```
    fn allclose<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        self.allclose_with(other, 1e-5, 1e-8)
    }

    /// Checks if all pairs of corresponding elements are close within user‑specified tolerances.
    ///
    /// The condition for closeness is: `|a - b| <= atol + rtol * |b|`.
    ///
    /// # Arguments
    /// * `other` – The other array (broadcastable shapes allowed).
    /// * `rtol` – Relative tolerance (default `1e-5` in `allclose`).
    /// * `atol` – Absolute tolerance (default `1e-8` in `allclose`).
    ///
    /// # Returns
    /// `Ok(true)` if all corresponding element pairs satisfy the above inequality,
    /// `Ok(false)` otherwise. Returns an error if shapes are not broadcastable.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, LogicOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![1.0001, 2.0001, 3.0001]);
    ///
    /// // With tight tolerances, not close:
    /// assert!(!a.allclose_with(&b, 1e-5, 1e-5).unwrap());
    ///
    /// // With looser tolerances, they become close:
    /// assert!(a.allclose_with(&b, 1e-3, 1e-3).unwrap());
    /// ```
    fn allclose_with<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
        rtol: f64,
        atol: f64,
    ) -> Result<bool, Self::Error> {
        if self.shape() == other.shape() {
            return Ok(self
                .iter()
                .zip(other.iter())
                .all(|(a, b)| (a - b).abs() <= atol + rtol * b.abs()));
        } else if other.ndim() == 0 {
            let scalar = match other.first() {
                Ok(&scalar) => scalar,
                Err(_) => return Err(ArrayError::ArrayNotAScalar),
            };
            return Ok(self
                .iter()
                .all(|a| (a - scalar).abs() <= atol + rtol * scalar.abs()));
        }

        let target_shape = match broadcast_shapes(self.shape(), other.shape()) {
            Ok(shape) => shape,
            Err(_) => return Err(ArrayError::IncompatibleShapes),
        };
        let total_len = target_shape.iter().product::<usize>();

        let strides_self = match broadcast_strides(self.shape(), self.strides(), &target_shape) {
            Ok(s) => s,
            Err(_) => return Err(ArrayError::IncompatibleShapes),
        };
        let strides_other = match broadcast_strides(other.shape(), other.strides(), &target_shape) {
            Ok(s) => s,
            Err(_) => return Err(ArrayError::IncompatibleShapes),
        };

        let offset_self = self.offset();
        let offset_other = other.offset();

        for flat_idx in 0..total_len {
            let mut rem = flat_idx;
            let mut idx_self = offset_self;
            let mut idx_other = offset_other;

            for dim in (0..target_shape.len()).rev() {
                let coord = rem % target_shape[dim];
                rem /= target_shape[dim];
                idx_self += coord * strides_self[dim];
                idx_other += coord * strides_other[dim];
            }

            // Safety: Using `unsafe` to avoid bounds checks - safe because indices are correct
            unsafe {
                let a = *self.data.as_ptr().add(idx_self);
                let b = *other.data().as_ptr().add(idx_other);
                if (a - b).abs() > atol + rtol * b.abs() {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Returns `true` if **all** corresponding element pairs are equal (element‑wise equality), after broadcasting.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, LogicOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert!(a.eq(&b).unwrap()); // all equal -> true
    ///
    /// let c = Array::from_vec(vec![1.0, 2.1, 3.0]);
    /// assert!(!a.eq(&c).unwrap()); // second element differs -> false
    ///
    /// // Scalar/Broadcasting: [1.0] becomes [1.0,1.0,1.0] -> not all equal
    /// let d = Array::from_vec(vec![1.0]);
    /// assert!(!a.eq(&d).unwrap());
    /// ```
    fn eq<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        compare_all_scalar(self, other, |a, b| a == b)
    }

    /// Returns `true` if **all** corresponding element pairs are unequal, after broadcasting.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, LogicOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![1.0, 2.1, 3.0]);
    /// assert!(!a.neq(&b).unwrap()); // only a single element differs -> false
    ///
    /// let d = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let e = Array::from_vec(vec![1.1, 2.1, 3.1]);
    /// assert!(d.neq(&e).unwrap()); // all pairs are unequal -> true
    /// ```
    fn neq<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        compare_all_scalar(self, other, |a, b| a != b)
    }

    /// Returns `true` if **all** corresponding elements satisfy `self > other`, after broadcasting.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, LogicOps};
    ///
    /// let a = Array::from_vec(vec![2.0, 3.0, 4.0]);
    /// let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert!(a.gt(&b).unwrap()); // all a[i] > b[i] -> true
    ///
    /// let c = Array::from_vec(vec![1.0, 2.0, 4.0]);
    /// assert!(!a.gt(&c).unwrap()); // last element 4.0 == 4.0 -> false
    /// ```
    fn gt<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        compare_all_scalar(self, other, |a, b| a > b)
    }

    /// Returns `true` if **all** corresponding elements satisfy `self < other`, after broadcasting.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, LogicOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![2.0, 3.0, 4.0]);
    /// assert!(a.lt(&b).unwrap()); // all a[i] < b[i] -> true
    ///
    /// let c = Array::from_vec(vec![2.0, 3.0, 3.0]);
    /// assert!(!a.lt(&c).unwrap()); // last element 3.0 == 3.0 -> false
    /// ```
    fn lt<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        compare_all_scalar(self, other, |a, b| a < b)
    }

    /// Returns `true` if **all** corresponding elements satisfy `self >= other`, after broadcasting.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, LogicOps};
    ///
    /// let a = Array::from_vec(vec![2.0, 2.0, 4.0]);
    /// let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert!(a.ge(&b).unwrap()); // all a[i] >= b[i] -> true
    ///
    /// let c = Array::from_vec(vec![2.0, 3.0, 4.0]);
    /// assert!(!a.ge(&c).unwrap()); // middle element !(2.0 >= 3.0) -> false
    /// ```
    fn ge<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        compare_all_scalar(self, other, |a, b| a >= b)
    }

    /// Returns `true` if **all** corresponding elements satisfy `self <= other`, after broadcasting.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, LogicOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![2.0, 2.0, 4.0]);
    /// assert!(a.le(&b).unwrap()); // all a[i] <= b[i] -> true
    ///
    /// let c = Array::from_vec(vec![1.0, 1.0, 4.0]);
    /// assert!(!a.le(&c).unwrap()); // middle element !(2.0 <= 1.0) -> false
    /// ```
    fn le<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        compare_all_scalar(self, other, |a, b| a <= b)
    }
}

fn compare_all_scalar<Rhs, F>(
    lhs: &ArrayView<'_, f64>,
    rhs: &Rhs,
    op: F,
) -> Result<bool, ArrayError>
where
    Rhs: AccessOps<Item = f64, Error = ArrayError>,
    F: Fn(f64, f64) -> bool,
{
    if lhs.shape() == rhs.shape() {
        return Ok(lhs.iter().zip(rhs.iter()).all(|(&a, &b)| op(a, b)));
    } else if rhs.ndim() == 0 {
        let scalar = match rhs.first() {
            Ok(&scalar) => scalar,
            Err(_) => return Err(ArrayError::ArrayNotAScalar),
        };
        return Ok(lhs.iter().all(|&a| op(a, scalar)));
    }

    let target_shape = match broadcast_shapes(lhs.shape(), rhs.shape()) {
        Ok(shape) => shape,
        Err(_) => return Err(ArrayError::IncompatibleShapes),
    };
    let total_len = target_shape.iter().product::<usize>();

    let strides_self = match broadcast_strides(lhs.shape(), lhs.strides(), &target_shape) {
        Ok(s) => s,
        Err(_) => return Err(ArrayError::IncompatibleShapes),
    };
    let strides_other = match broadcast_strides(rhs.shape(), rhs.strides(), &target_shape) {
        Ok(s) => s,
        Err(_) => return Err(ArrayError::IncompatibleShapes),
    };

    let offset_self = lhs.offset();
    let offset_other = rhs.offset();

    for flat_idx in 0..total_len {
        let mut rem = flat_idx;
        let mut idx_self = offset_self;
        let mut idx_other = offset_other;

        for dim in (0..target_shape.len()).rev() {
            let coord = rem % target_shape[dim];
            rem /= target_shape[dim];
            idx_self += coord * strides_self[dim];
            idx_other += coord * strides_other[dim];
        }

        // Safety: Using `unsafe` to avoid bounds checks - safe because indices are correct
        unsafe {
            let a = *lhs.data.as_ptr().add(idx_self);
            let b = *rhs.data().as_ptr().add(idx_other);
            if !op(a, b) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

impl LogicOps for Array<f64, Vec<f64>> {
    /// Returns `true` if all elements of the array are finite (not infinite and not NaN).
    ///
    /// See [`ArrayView::is_finite`] for details.
    fn is_finite(&self) -> bool {
        self.view().is_finite()
    }

    /// Returns `true` if all elements of the array are infinite (either positive or negative infinity).
    ///
    /// See [`ArrayView::is_inf`] for details.
    fn is_inf(&self) -> bool {
        self.view().is_inf()
    }

    /// Returns `true` if all elements of the array are NaN.
    ///
    /// See [`ArrayView::is_nan`] for details.
    fn is_nan(&self) -> bool {
        self.view().is_nan()
    }

    /// Returns `true` if all corresponding elements are close within absolute and relative tolerances.
    ///
    /// See [`ArrayView::allclose`] for details.
    fn allclose<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        self.view().allclose(other)
    }

    /// Checks if all pairs of corresponding elements are close within user‑specified tolerances.
    ///
    /// See [`ArrayView::allclose_with`] for details.
    fn allclose_with<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
        rtol: f64,
        atol: f64,
    ) -> Result<bool, Self::Error> {
        self.view().allclose_with(other, rtol, atol)
    }

    /// Returns `true` if **all** corresponding element pairs are equal (element‑wise equality), after broadcasting.
    ///
    /// See [`ArrayView::eq`] for details.
    fn eq<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        self.view().eq(other)
    }

    /// Returns `true` if **all** corresponding element pairs are unequal, after broadcasting.
    ///
    /// See [`ArrayView::neq`] for details.
    fn neq<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        self.view().neq(other)
    }

    /// Returns `true` if **all** corresponding elements satisfy `self > other`, after broadcasting.
    ///
    /// See [`ArrayView::gt`] for details.
    fn gt<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        self.view().gt(other)
    }

    /// Returns `true` if **all** corresponding elements satisfy `self < other`, after broadcasting.
    ///
    /// See [`ArrayView::lt`] for details.
    fn lt<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        self.view().lt(other)
    }

    /// Returns `true` if **all** corresponding elements satisfy `self >= other`, after broadcasting.
    ///
    /// See [`ArrayView::ge`] for details.
    fn ge<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        self.view().ge(other)
    }

    /// Returns `true` if **all** corresponding elements satisfy `self <= other`, after broadcasting.
    ///
    /// See [`ArrayView::le`] for details.
    fn le<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error> {
        self.view().le(other)
    }
}
