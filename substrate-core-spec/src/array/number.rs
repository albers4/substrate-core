// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use crate::{Add, Div, Mul, Neg, Rem, Sub};

pub trait Zero {
    fn zero() -> Self;
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0f32
    }
}
impl Zero for f64 {
    fn zero() -> Self {
        0.0f64
    }
}

pub trait One {
    fn one() -> Self;
}

impl One for f32 {
    fn one() -> Self {
        1.0f32
    }
}
impl One for f64 {
    fn one() -> Self {
        1.0f64
    }
}

pub trait NumberOps<Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Div<Rhs, Output = Output>
    + Rem<Rhs, Output = Output>
    + Neg<Output = Output>
{
}

impl NumberOps for f32 {}
impl NumberOps for f64 {}

pub trait Number: PartialEq + Zero + One + NumberOps + Sized {}

impl Number for f32 {}
impl Number for f64 {}
