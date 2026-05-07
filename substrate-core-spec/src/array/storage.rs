// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use crate::array::number::Number;
use std::vec::Vec;

pub trait Storage {
    type Item: Number;

    fn as_slice(&self) -> &[Self::Item];
    fn as_mut_slice(&mut self) -> &mut [Self::Item];
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Storage for Vec<f64> {
    type Item = f64;

    fn as_slice(&self) -> &[Self::Item] {
        self
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Item] {
        self
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<const N: usize> Storage for [f64; N] {
    type Item = f64;

    fn as_slice(&self) -> &[Self::Item] {
        self
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Item] {
        self
    }

    fn len(&self) -> usize {
        N
    }
}
