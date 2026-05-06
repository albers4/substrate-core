// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use core::ops::Range;

use crate::Vec;
use crate::array::ArrayViewLike;
use crate::array::pad_mode::PadMode;
use crate::array::{ArrayLike, index::ToIndex};

pub trait InitOps: ArrayLike + Sized {
    type Output: ArrayLike;

    /// Uniform random numbers in [0,1).
    fn rand(shape: &[usize]) -> Result<Self::Output, Self::Error>;
    /// Normal (Gaussian) random numbers.
    fn randn(shape: &[usize]) -> Result<Self::Output, Self::Error>;
    /// Identity matrix (2-D only).
    fn eye(n: usize) -> Result<Self::Output, Self::Error>;
    /// Diagonal matrix from a 1-D array.
    fn diag(
        diag: &impl AccessOps<Item = Self::Item, Error = Self::Error>,
    ) -> Result<Self::Output, Self::Error>;
    /// Fill array with a constant value.
    fn full(shape: &[usize], value: Self::Item) -> Result<Self::Output, Self::Error>;
    /// Regulary spaced 1-D array.
    fn arange(
        start: Self::Item,
        end: Self::Item,
        step: Self::Item,
    ) -> Result<Self::Output, Self::Error>;
    /// Log-spaced array.
    fn logspace(a: Self::Item, b: Self::Item, n: usize) -> Result<Self::Output, Self::Error>;
    /// Create array of zeros with given shape.
    fn zeros(shape: &[usize]) -> Self::Output;
    /// Create array of ones with given shape.
    fn ones(shape: &[usize]) -> Self::Output;
    /// Create a 1-D array from a `Vec`.
    fn from_vec(vec: Vec<Self::Item>) -> Self::Output;
    /// Create a 1-D array with `n` linearly spaced elements from `a` to `b`.
    fn linspace(a: Self::Item, b: Self::Item, n: usize) -> Result<Self::Output, Self::Error>;
    /// Create an array by calling a closure for each logical index.
    fn from_fn<F>(shape: &[usize], f: F) -> Result<Self::Output, Self::Error>
    where
        F: FnMut(&[usize]) -> Self::Item;
}

pub trait AccessOps: ArrayLike + Sized {
    type Output: ArrayLike;
    type View<'a>: ArrayLike
    where
        Self: 'a;

    /// Returns the underlying data slice.
    fn data(&self) -> &[Self::Item];

    /// # Safety
    /// Unsafely returns a reference to the element at `indices` without bounds checking.
    unsafe fn get_unchecked<I: ToIndex>(&self, indices: &[I]) -> &Self::Item
    where
        I::Error: core::fmt::Debug;
    /// Returns a reference to the first logical element.
    fn first(&self) -> Result<&Self::Item, Self::Error>;
    /// Returns a reference to the last logical element.
    fn last(&self) -> Result<&Self::Item, Self::Error>;
    /// Returns a reference to the element at the given flat logical index.
    fn get_flat(&self, index: impl ToIndex) -> Result<&Self::Item, Self::Error>;
    /// Returns a reference to the element at the given multi-dimensional indices.
    fn get(&self, indices: &[impl ToIndex]) -> Result<&Self::Item, Self::Error>;
    /// Returns a scalar view by indexing with one index per axis.
    fn slice_by_indices(&self, indices: &[impl ToIndex]) -> Result<Self::View<'_>, Self::Error>;
    /// Returns a view sliced along one axis with a contiguous range.
    fn slice_by_range(
        &self,
        axis: usize,
        range: Range<usize>,
    ) -> Result<Self::View<'_>, Self::Error>;
    /// Returns a view sliced along one axis with a start, end and step.
    fn slice_by_stride(
        &self,
        axis: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Self::View<'_>, Self::Error>;
    /// Returns a view where elemenets are selected by a boolean mask.
    fn select(&self, mask: &Self::View<'_>) -> Result<Self::Output, Self::Error>;
    /// Returns a view taking elements at given flat indices.
    fn take(&self, indices: &Self::View<'_>) -> Result<Self::Output, Self::Error>;
    /// Returns a view gathering elements from a coordinate list.
    fn gather(
        &self,
        dim: impl ToIndex,
        indices: &Self::View<'_>,
    ) -> Result<Self::Output, Self::Error>;
    /// Returns an iterator over logical elements in memory order.
    fn iter(&self) -> impl Iterator<Item = &Self::Item>;
}

pub trait AccessOpsMut: AccessOps {
    /// Returns a mutable reference to the elemenet at the given flat logical index.
    fn get_flat_mut(&mut self, index: impl ToIndex) -> Result<&mut Self::Item, Self::Error>;
    /// # Safety
    /// Unsafely returns a mutable reference to the element at the given flat logical index.
    unsafe fn get_unchecked_mut<I: ToIndex>(&mut self, indices: &[I]) -> &mut Self::Item
    where
        I::Error: core::fmt::Debug;
    /// Returns a mutable reference the the element at the given multi-dimensional indices.
    fn get_mut(&mut self, indices: &[impl ToIndex]) -> Result<&mut Self::Item, Self::Error>;
    /// Returns a mutable iterator over logical elements in memory order.
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item>;
    /// Sets the element at the given flat logical index to `value`.
    fn set_flat(&mut self, index: impl ToIndex, value: Self::Item) -> Result<(), Self::Error>;
    /// Sets the element at the given multi-dimensional indices to `value`.
    fn set(&mut self, indices: &[impl ToIndex], value: Self::Item) -> Result<(), Self::Error>;
    /// # Safety
    /// Unsafely sets the element at `indices` to `value` without bounds checking.
    unsafe fn set_unchecked(
        &mut self,
        indices: &[impl ToIndex],
        value: Self::Item,
    ) -> Result<(), Self::Error>;
}

pub trait UnaryOps: ArrayLike {
    type Output: ArrayLike;

    /// Element-wise absolute value.
    fn abs(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise negation.
    fn neg(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise root.
    fn sqrt(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise exponential (e^x).
    fn exp(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise natural logarithm.
    fn ln(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise logarithm with given base.
    fn log(&self, base: f64) -> Result<Self::Output, Self::Error>;
    /// Element-wise sine (radians).
    fn sin(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise cosine (radians).
    fn cos(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise tangent (radians).
    fn tan(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise arcsine.
    fn asin(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise arccosine.
    fn acos(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise arctangent.
    fn atan(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise hyperbolic sine.
    fn sinh(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise hyperbolic cosine.
    fn cosh(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise hyperbolic tangent.
    fn tanh(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise ceiling (smallest integer >= value).
    fn ceil(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise floor (largest integer <= value).
    fn floor(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise rounding to nearest integer.
    fn round(&self) -> Result<Self::Output, Self::Error>;
    /// Element-wise signum (-1, 1).
    fn signum(&self) -> Result<Self::Output, Self::Error>;
}

pub trait BinaryOps: ArrayLike {
    type Output: ArrayLike;

    /// Element-wise addition with broadcasting.
    fn add<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error>;
    /// Element-wise subtraction with broadcasting.
    fn sub<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error>;
    /// Element-wise multiplication with broadcasting.
    fn mul<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error>;
    /// Element-wise division with broadcasting.
    fn div<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error>;
    /// Element-wise power (array ^ array) with broadcasting.
    fn pow<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error>;
    /// Element-wise remainder (modulus) with broadcasting.
    fn rem<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error>;
    /// Element-wise maximum with broadcasting.
    fn max<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error>;
    /// Element-wise minimum with broadcasting.
    fn min<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error>;
}

pub trait ConvertOps: ArrayLike {
    type View<'a>: ArrayViewLike
    where
        Self: 'a;

    /// Returns the single element as a scalar.
    fn to_scalar(&self) -> Result<Self::Item, Self::Error>;
    /// Copies all elements into a new `Vec`.
    fn to_vec(&self) -> Vec<Self::Item>;
    /// Creates a borrowed view of the entire array.
    fn view(&self) -> Self::View<'_>;
}

pub trait ReduceOps: ArrayLike {
    type Output: ArrayLike;

    /// Sum of all elements (scalar result).
    fn sum(&self) -> Result<Self::Output, Self::Error>;
    /// Sum along the given axis (reduces dimension by one).
    fn sum_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    /// Arithmetic mean of all elements.
    fn mean(&self) -> Result<Self::Output, Self::Error>;
    /// Mean along the given axis.
    fn mean_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    /// Variance of all elements (sample variance).
    fn var(&self) -> Result<Self::Output, Self::Error>;
    /// Variance along the given axis.
    fn var_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    /// Standard deviation of all elements.
    fn std(&self) -> Result<Self::Output, Self::Error>;
    /// Standard deviation along the given axis.
    fn std_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    /// Product of all elements.
    fn prod(&self) -> Result<Self::Output, Self::Error>;
    /// Product along the given axis.
    fn prod_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    /// Minimum value among all elements.
    fn min(&self) -> Result<Self::Output, Self::Error>;
    /// Minimum value along the given axis.
    fn min_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    /// Maximum value among all elements.
    fn max(&self) -> Result<Self::Output, Self::Error>;
    /// Maximum value along the given axis.
    fn max_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    /// Index of the minimum value (flattened).
    fn argmin(&self) -> Result<Self::Output, Self::Error>;
    /// Index of the minimum along the given axis.
    fn argmin_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    /// Index of the maximum value (flattened).
    fn argmax(&self) -> Result<Self::Output, Self::Error>;
    /// Index of the maximum along the given axis.
    fn argmax_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    /// Are any elements equal to `value`? (scalar boolean result).
    fn any(&self, value: Self::Item) -> Result<Self::Output, Self::Error>;
    /// Any elements along axis equals `value`? (reduces dimension by one).
    fn any_axis(&self, value: Self::Item, axis: usize) -> Result<Self::Output, Self::Error>;
    /// Do all elements equal `value`?
    fn all(&self, value: Self::Item) -> Result<Self::Output, Self::Error>;
    /// All elements along axis equal `value`?
    fn all_axis(&self, value: Self::Item, axis: usize) -> Result<Self::Output, Self::Error>;
}

pub trait LinearAlgebraOps: ArrayLike {
    type Output: ArrayLike;
    type View<'a>: ArrayViewLike + 'a
    where
        Self: 'a;

    /// Dot product of two 1-D arrays (also known as inner product).
    fn dot<Rhs: AccessOps>(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    /// Matrix multiplication (2-D arrays), support broadcasting.
    fn matmul<Rhs: AccessOps>(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    /// Lazy transpose (view) - swaps shape and strides, O(1).
    fn transpose(&self) -> Result<Self::View<'_>, Self::Error>;
    /// Eager transpose - copies data to a new contiguous array.
    fn transpose_copy(&self) -> Result<Self::Output, Self::Error>;
    /// Trace (sum of diagonal elements) of a 2-D array.
    fn trace(&self) -> Result<Self::Output, Self::Error>;
    /// Determinant of a square matrix.
    fn det(&self) -> Result<Self::Output, Self::Error>;
    /// Inverse of a square matrix.
    fn inv(&self) -> Result<Self::Output, Self::Error>;
    /// Solves a linear system.
    fn solve(&self) -> Result<Self::Output, Self::Error>;
    /// Eigenvalues and eigenvectors.
    fn eig(&self) -> Result<Self::Output, Self::Error>;
    /// Singular Value Decomposition.
    fn svd(&self) -> Result<Self::Output, Self::Error>;
    /// QR decomposition.
    fn qr(&self) -> Result<Self::Output, Self::Error>;
    /// Cholesky decomposition.
    fn cholesky(&self) -> Result<Self::Output, Self::Error>;
    /// Matrix norm (Frobenius or L2).
    fn norm(&self) -> Result<Self::Output, Self::Error>;
    /// Cross product of two 3-D vectors.
    fn cross<Rhs: AccessOps>(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
}

pub trait LogicOps: ArrayLike {
    /// Are all elements finite?
    fn is_finite(&self) -> bool;
    /// Are all elements infinite?
    fn is_inf(&self) -> bool;
    /// Are all elements NaN?
    fn is_nan(&self) -> bool;
    /// Are all elements close (withing tolerance, rtol=1e-5, atol=1e-8)?
    fn allclose<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error>;
    /// Are all elements close (withing tolerance)?
    fn allclose_with<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
        rtol: f64,
        atol: f64,
    ) -> Result<bool, Self::Error>;
    /// Element-wise equality with broadcasting.
    fn eq<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error>;
    /// Element-wise inequality with broadcasting.
    fn neq<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error>;
    /// Element-wise greater-than with broadcasting.
    fn gt<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error>;
    /// Element-wise less-than with broadcasting.
    fn lt<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error>;
    /// Element-wise greater-than-or-equal with broadcasting.
    fn ge<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error>;
    /// Element-wise less-than-or-equal with broadcasting.
    fn le<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<bool, Self::Error>;
}

pub trait ShapeOps: ArrayLike {
    type Output: ArrayLike;
    type View<'a>: ArrayViewLike + 'a
    where
        Self: 'a;

    /// Zero-copy reshape of contiguous, error otherwise.
    fn reshape(&self, new_shape: &[usize]) -> Result<Self::View<'_>, Self::Error>;
    /// Reshape, making a contiguous copy if necessary.
    fn reshape_copy(self, new_shape: &[usize]) -> Result<Self::Output, Self::Error>;
    /// Convert to contiguous row-major (C order).
    fn to_row_major(self) -> Result<Self::Output, Self::Error>;
    /// Convert to contiguous column-major (Fortran order).
    fn to_column_major(self) -> Result<Self::Output, Self::Error>;
    /// Flatten to 1-D (row-major).
    fn flatten(&self) -> Result<Self::Output, Self::Error>;
    /// Removes axes of size 1.
    fn squeeze(&self) -> Result<Self::Output, Self::Error>;
    /// Add a new axis of size 1.
    fn unsqueeze(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    /// Broadcast to a new shape.
    fn broadcast_to(&self, shape: &[usize]) -> Result<Self::Output, Self::Error>;
    /// Concatenate arrays along an axis.
    fn concatenate(
        &self,
        arrays: &[Self::View<'_>],
        axis: usize,
    ) -> Result<Self::Output, Self::Error>;
    /// Stack arrays along a new axis.
    fn stack(&self, arrays: &[Self::View<'_>], axis: usize) -> Result<Self::Output, Self::Error>;
    /// Split array into multiple sub-arrays.
    fn split(
        &self,
        indices_or_sections: usize,
        axis: usize,
    ) -> Result<Vec<Self::Output>, Self::Error>;
    /// Roll elements along an axis.
    fn roll(&self, shift: isize, axis: Option<usize>) -> Result<Self::Output, Self::Error>;
    /// Pad array with constant/edge values.
    fn pad(&self, pad_width: &[(usize, usize)], mode: PadMode)
    -> Result<Self::Output, Self::Error>;
    /// Repeat array by tiling.
    fn tile(&self, reps: &[usize]) -> Result<Self::Output, Self::Error>;
}
