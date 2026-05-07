// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::marker::PhantomData;

use crate::semantics::{arithmetic::Arithmetic, dimension::Dimension};

#[derive(PartialEq, Eq, Debug)]
pub struct Quantity<T: Arithmetic, D: Dimension> {
    pub(crate) value: T,
    _dim: PhantomData<D>,
}

impl<T: Arithmetic, D: Dimension> Quantity<T, D> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            _dim: PhantomData,
        }
    }

    pub fn value(self) -> T {
        self.value
    }

    pub fn dim_string(&self) -> &'static str {
        D::to_string()
    }
}
