// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::ops::{Add, Div, Mul, Sub};

use crate::semantics::{
    arithmetic::Arithmetic,
    dimension::{Dimension, DivDim, MulDim},
    quantity::Quantity,
};

impl<T: Arithmetic, D: Dimension> Add<Self> for Quantity<T, D> {
    type Output = Quantity<T, D>;

    fn add(self, rhs: Self) -> Self::Output {
        let new_value = self.value + rhs.value;
        Quantity::new(new_value)
    }
}

impl<T: Arithmetic, D: Dimension> Sub<Self> for Quantity<T, D> {
    type Output = Quantity<T, D>;

    fn sub(self, rhs: Self) -> Self::Output {
        let new_value = self.value - rhs.value;
        Quantity::new(new_value)
    }
}

impl<T: Arithmetic, D1: Dimension, D2: Dimension> Mul<Quantity<T, D2>> for Quantity<T, D1> {
    type Output = Quantity<T, MulDim<D1, D2>>;

    fn mul(self, rhs: Quantity<T, D2>) -> Self::Output {
        let new_value = self.value * rhs.value;
        Quantity::new(new_value)
    }
}

impl<T: Arithmetic, D1: Dimension, D2: Dimension> Div<Quantity<T, D2>> for Quantity<T, D1> {
    type Output = Quantity<T, DivDim<D1, D2>>;

    fn div(self, rhs: Quantity<T, D2>) -> Self::Output {
        let new_value = self.value / rhs.value;
        Quantity::new(new_value)
    }
}
