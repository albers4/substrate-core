// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::ops::{Add, Div, Mul, Sub};

pub trait Arithmetic:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Copy
    + PartialOrd
{
}

impl Arithmetic for f64 {}
impl Arithmetic for f32 {}
