// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

pub mod index;
pub mod memory_order;
pub mod pad_mode;
pub mod number;
pub mod ops;
pub mod storage;

use memory_order::MemoryOrder;

use crate::array::{index::ToIndex, number::Number};

pub trait ArrayLike {
    type Item: Number;
    type Error;

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

    fn physical_from_indices(&self, indices: &[impl ToIndex]) -> Result<usize, Self::Error>;
    fn physical_from_logical_flat(&self, index: usize) -> Result<usize, Self::Error>;
}

pub trait ArrayViewLike: ArrayLike + Send + Sync {
    type Owned: ArrayLike;

    fn into_owned(self) -> Self::Owned;
}
