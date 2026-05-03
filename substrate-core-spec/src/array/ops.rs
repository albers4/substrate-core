// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use core::ops::Range;

use crate::Vec;
use crate::array::ArrayViewLike;
use crate::array::{ArrayLike, index::ToIndex, number::Number};

pub trait AccessOps: Sized {
    type Item: Number;
    type Output: ArrayLike;
    type Error;

    /// # Safety
    unsafe fn get_unchecked<I: ToIndex>(&self, indices: &[I]) -> &Self::Item
    where
        I::Error: core::fmt::Debug;
    fn first(&self) -> Result<&Self::Item, Self::Error>;
    fn last(&self) -> Result<&Self::Item, Self::Error>;
    fn get_flat(&self, index: impl ToIndex) -> Result<&Self::Item, Self::Error>;

    fn get(&self, indices: &[impl ToIndex]) -> Result<&Self::Item, Self::Error>;
    fn slice_by_indices(&self, indices: &[impl ToIndex]) -> Result<Self, Self::Error>;
    fn slice_by_range(&self, axis: usize, range: Range<usize>) -> Result<Self, Self::Error>;
    fn slice_by_stride(
        &self,
        axis: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Self, Self::Error>;

    /// using boolean mask
    fn select(&self) -> Result<Self::Output, Self::Error>;
    fn take(&self) -> Result<Self::Output, Self::Error>;
    /// by coordinate list
    fn gather(&self) -> Result<Self::Output, Self::Error>;
    fn iter(&self) -> impl Iterator<Item = Self::Item>;
}

pub trait AccessOpsMut: AccessOps {
    fn get_flat_mut(&mut self, index: impl ToIndex) -> Result<&mut Self::Item, Self::Error>;
    /// Safety
    unsafe fn get_unchecked_mut<I: ToIndex>(&mut self, indices: &[I]) -> &mut Self::Item
    where
        I::Error: core::fmt::Debug;

    fn get_mut(&mut self, indices: &[impl ToIndex]) -> Result<&mut Self::Item, Self::Error>;
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item>;

    fn set_flat(&mut self, index: impl ToIndex, value: Self::Item) -> Result<(), Self::Error>;
    fn set(&mut self, indices: &[impl ToIndex], value: Self::Item) -> Result<(), Self::Error>;
    /// # Safety
    unsafe fn set_unchecked(
        &mut self,
        indices: &[impl ToIndex],
        value: Self::Item,
    ) -> Result<(), Self::Error>;
}

pub trait UnaryOps {
    type Output: ArrayLike;
    type Error;

    fn abs(&self) -> Result<Self::Output, Self::Error>;
    fn neg(&self) -> Result<Self::Output, Self::Error>;
    fn sqrt(&self) -> Result<Self::Output, Self::Error>;
    fn exp(&self) -> Result<Self::Output, Self::Error>;
    fn ln(&self) -> Result<Self::Output, Self::Error>;
    fn log(&self, base: usize) -> Result<Self::Output, Self::Error>;
    fn sin(&self) -> Result<Self::Output, Self::Error>;
    fn cos(&self) -> Result<Self::Output, Self::Error>;
    fn tan(&self) -> Result<Self::Output, Self::Error>;
    fn asin(&self) -> Result<Self::Output, Self::Error>;
    fn acos(&self) -> Result<Self::Output, Self::Error>;
    fn atan(&self) -> Result<Self::Output, Self::Error>;
    fn sinh(&self) -> Result<Self::Output, Self::Error>;
    fn cosh(&self) -> Result<Self::Output, Self::Error>;
    fn tanh(&self) -> Result<Self::Output, Self::Error>;
    fn ceil(&self) -> Result<Self::Output, Self::Error>;
    fn floor(&self) -> Result<Self::Output, Self::Error>;
    fn round(&self) -> Result<Self::Output, Self::Error>;
    fn signum(&self) -> Result<Self::Output, Self::Error>;
}

pub trait BinaryOps<Rhs = Self> {
    type Output: ArrayLike;
    type Error;

    fn add(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    fn sub(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    fn mul(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    fn div(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    fn pow(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    fn rem(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    fn max(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    fn min(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
}

pub trait ConvertOps {
    type Item: Number;
    type Error;

    fn to_scalar(&self) -> Result<Self::Item, Self::Error>;
    fn to_vec(&self) -> Vec<Self::Item>;
}

pub trait ReduceOps {
    type Output: ArrayLike;
    type Error;

    fn sum(&self) -> Result<Self::Output, Self::Error>;
    fn sum_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    fn mean(&self) -> Result<Self::Output, Self::Error>;
    fn mean_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    fn var(&self) -> Result<Self::Output, Self::Error>;
    fn var_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    fn std(&self) -> Result<Self::Output, Self::Error>;
    fn std_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    fn prod(&self) -> Result<Self::Output, Self::Error>;
    fn prod_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    fn min(&self) -> Result<Self::Output, Self::Error>;
    fn min_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    fn max(&self) -> Result<Self::Output, Self::Error>;
    fn max_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    fn argmin(&self) -> Result<Self::Output, Self::Error>;
    fn argmin_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    fn argmax(&self) -> Result<Self::Output, Self::Error>;
    fn argmax_axis(&self, axis: usize) -> Result<Self::Output, Self::Error>;
    fn any(&self, value: impl Number) -> Result<Self::Output, Self::Error>;
    fn any_axis(&self, value: impl Number, axis: usize) -> Result<Self::Output, Self::Error>;
    fn all(&self, value: impl Number) -> Result<Self::Output, Self::Error>;
    fn all_axis(&self, value: impl Number, axis: usize) -> Result<Self::Output, Self::Error>;
}

pub trait LinearAlgebraOps<Rhs = Self> {
    type Output: ArrayLike;
    type Error;
    type View<'a>: ArrayViewLike + 'a
    where
        Self: 'a;

    fn dot(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    fn matmul(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;

    fn transpose(&self) -> Result<Self::View<'_>, Self::Error>;
    fn transpose_copy(&self) -> Result<Self::Output, Self::Error>;

    fn trace(&self) -> Result<Self::Output, Self::Error>;
    fn det(&self) -> Result<Self::Output, Self::Error>;
    fn inv(&self) -> Result<Self::Output, Self::Error>;
    fn solve(&self) -> Result<Self::Output, Self::Error>;
    fn eig(&self) -> Result<Self::Output, Self::Error>;
    fn svd(&self) -> Result<Self::Output, Self::Error>;
    fn qr(&self) -> Result<Self::Output, Self::Error>;
    fn cholesky(&self) -> Result<Self::Output, Self::Error>;
    fn norm(&self) -> Result<Self::Output, Self::Error>;
    fn cross(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
}

pub trait LogicOps<Rhs = Self> {
    type Output: ArrayLike;
    type Error;

    fn is_finite(&self) -> Result<Self::Output, Self::Error>;
    fn is_inf(&self) -> Result<Self::Output, Self::Error>;
    fn is_nan(&self) -> Result<Self::Output, Self::Error>;
    fn allclose(&self) -> Result<Self::Output, Self::Error>;
    fn eq(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    fn neq(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    fn gt(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    fn lt(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    fn ge(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
    fn le(&self, other: &Rhs) -> Result<Self::Output, Self::Error>;
}

pub trait ShapeOps {
    type Output: ArrayLike;
    type Error;

    fn reshape(self, new_shape: &[usize]) -> Result<Self::Output, Self::Error>;
    fn reshape_copy(&self, new_shape: &[usize]) -> Result<Self::Output, Self::Error>;
    fn into_shape(self, new_shape: &[usize]) -> Result<Self::Output, Self::Error>;

    fn to_row_major(self) -> Result<Self::Output, Self::Error>;
    fn to_column_major(self) -> Result<Self::Output, Self::Error>;

    fn flatten(&self) -> Result<Self::Output, Self::Error>;
    fn squeeze(&self) -> Result<Self::Output, Self::Error>;
    fn unsqueeze(&self) -> Result<Self::Output, Self::Error>;
    fn broadcast_to(&self) -> Result<Self::Output, Self::Error>;
    fn concatenate(&self) -> Result<Self::Output, Self::Error>;
    fn stack(&self) -> Result<Self::Output, Self::Error>;
    fn split(&self) -> Result<Self::Output, Self::Error>;
    fn roll(&self) -> Result<Self::Output, Self::Error>;
    fn pad(&self) -> Result<Self::Output, Self::Error>;
    fn tile(&self) -> Result<Self::Output, Self::Error>;
}

pub trait InitOps: Sized {
    type Item: Number;
    type Output: ArrayLike;
    type Error;

    /// Uniform
    fn rand();
    /// Normal
    fn randn();
    fn eye();
    fn diag();
    fn full();
    fn arange();
    fn logspace();

    fn zeros(shape: &[usize]) -> Self;
    fn ones(shape: &[usize]) -> Self;
    fn from_vec(vec: Vec<Self::Item>) -> Self;
    fn linspace(a: Self::Item, b: Self::Item, n: usize) -> Result<Self, Self::Error>;
    fn from_fn<F>(shape: &[usize], f: F) -> Result<Self, Self::Error>
    where
        F: FnMut(&[usize]) -> Self::Item;
}