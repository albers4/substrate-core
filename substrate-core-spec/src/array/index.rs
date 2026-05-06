// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use core::num::TryFromIntError;

pub trait ToIndex: Copy {
    type Error: core::fmt::Debug;

    fn to_index(self) -> Result<usize, Self::Error>;
}

impl ToIndex for usize {
    type Error = core::convert::Infallible;

    fn to_index(self) -> Result<usize, Self::Error> {
        Ok(self)
    }
}

impl ToIndex for i32 {
    type Error = TryFromIntError;

    fn to_index(self) -> Result<usize, Self::Error> {
        self.try_into()
    }
}

impl ToIndex for i64 {
    type Error = TryFromIntError;

    fn to_index(self) -> Result<usize, Self::Error> {
        self.try_into()
    }
}

#[derive(Debug)]
pub enum FloatToIndexError {
    Negative,
    NotAnInteger,
    IndexOutOfBounds,
}

impl ToIndex for f64 {
    type Error = FloatToIndexError;

    fn to_index(self) -> Result<usize, Self::Error> {
        if self.is_sign_negative() {
            return Err(FloatToIndexError::Negative);
        }
        if self.fract() != 0.0 {
            return Err(FloatToIndexError::NotAnInteger);
        }
        if self > (usize::MAX as f64) {
            return Err(FloatToIndexError::IndexOutOfBounds);
        }
        Ok(self as usize)
    }
}

impl ToIndex for f32 {
    type Error = FloatToIndexError;

    fn to_index(self) -> Result<usize, Self::Error> {
        if self.is_sign_negative() {
            return Err(FloatToIndexError::Negative);
        }
        if self.fract() != 0.0 {
            return Err(FloatToIndexError::NotAnInteger);
        }
        if self > (usize::MAX as f32) {
            return Err(FloatToIndexError::IndexOutOfBounds);
        }
        Ok(self as usize)
    }
}
