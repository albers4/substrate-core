// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use crate::TryFromIntError;

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
