// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

pub mod index;
pub mod memory_order;
pub mod number;
pub mod storage;
pub mod ops;

use index::ToIndex;
use memory_order::MemoryOrder;
use number::Number;

use crate::Range;
use crate::Vec;

pub trait ArrayAccess {
    type Item: Number;
    type Error;

    // Access
    fn get_flat(&self, index: impl ToIndex) -> Result<&Self::Item, Self::Error>;
    fn get(&self, indices: &[impl ToIndex]) -> Result<&Self::Item, Self::Error>;
    /// # Safety
    unsafe fn get_unchecked<I: ToIndex>(&self, indices: &[I]) -> &Self::Item
    where
        I::Error: core::fmt::Debug;
    fn first(&self) -> Result<&Self::Item, Self::Error>;
    fn last(&self) -> Result<&Self::Item, Self::Error>;

    // Iteration
    fn iter(&self) -> impl Iterator<Item = Self::Item>;

    // Shape & Layout
    fn length(&self) -> usize;
    fn storage_length(&self) -> usize;
    fn size(&self) -> usize;
    fn ndim(&self) -> usize;
    fn order(&self) -> MemoryOrder;
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn offset(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn is_contiguous(&self) -> bool;
    fn is_canonical(&self, order: MemoryOrder) -> bool;

    // Conversion
    fn to_scalar(&self) -> Result<Self::Item, Self::Error>;
    fn to_vec(&self) -> Vec<Self::Item>;

    // Indexing
    fn physical_from_indices(&self, indices: &[impl ToIndex]) -> Result<usize, Self::Error>;
    fn physical_from_logical_flat(&self, index: usize) -> Result<usize, Self::Error>;
}

pub trait ArrayAccessMut: ArrayAccess {
    // Access
    fn get_flat_mut(&mut self, index: impl ToIndex) -> Result<&mut Self::Item, Self::Error>;
    fn get_mut(&mut self, indices: &[impl ToIndex]) -> Result<&mut Self::Item, Self::Error>;
    /// # Safety
    unsafe fn get_unchecked_mut<I: ToIndex>(&mut self, indices: &[I]) -> &mut Self::Item
    where
        I::Error: core::fmt::Debug;

    // Iteration
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Item>;
}

pub trait ArrayLike: ArrayAccess + Sized {
    type View<'a>: ArrayViewLike + 'a
    where
        Self: 'a;

    // Creation
    fn zeros(shape: &[usize]) -> Self;
    fn ones(shape: &[usize]) -> Self;
    fn from_vec(vec: Vec<Self::Item>) -> Self;
    fn linspace(a: Self::Item, b: Self::Item, n: usize) -> Result<Self, Self::Error>;
    fn from_fn<F>(shape: &[usize], f: F) -> Result<Self, Self::Error>
    where
        F: FnMut(&[usize]) -> Self::Item;

    // Transpose
    fn transpose(&self) -> Result<Self::View<'_>, Self::Error>;
    fn transpose_copy(&self) -> Result<Self, Self::Error>;

    // Reshaping
    fn reshape(self, new_shape: &[usize]) -> Result<Self, Self::Error>;
    fn into_shape(self, new_shape: &[usize]) -> Result<Self, Self::Error>;

    // Reordering
    fn to_row_major(self) -> Result<Self, Self::Error>;
    fn to_column_major(self) -> Result<Self, Self::Error>;

    // View creation
    fn view(&self) -> Self::View<'_>;
}

pub trait ArrayLikeMut: ArrayLike + ArrayAccessMut {
    // Set
    fn set_flat(&mut self, index: impl ToIndex, value: Self::Item) -> Result<(), Self::Error>;
    fn set(&mut self, indices: &[impl ToIndex], value: Self::Item) -> Result<(), Self::Error>;
    /// # Safety
    unsafe fn set_unchecked(
        &mut self,
        indices: &[impl ToIndex],
        value: Self::Item,
    ) -> Result<(), Self::Error>;
}

pub trait ArrayViewLike: ArrayAccess + Sized + Send + Sync {
    type Owned: ArrayLike;

    fn into_owned(self) -> Self::Owned;
    fn slice_by_indices(&self, indices: &[impl ToIndex]) -> Result<Self, Self::Error>;
    fn slice_by_range(&self, axis: usize, range: Range<usize>) -> Result<Self, Self::Error>;
    fn slice_by_stride(
        &self,
        axis: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Self, Self::Error>;
}
