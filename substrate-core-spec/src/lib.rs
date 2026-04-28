// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#![no_std]

/// Specification: v1.0
extern crate alloc;
use alloc::vec::Vec;
use core::num::TryFromIntError;
use core::ops::Range;
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

pub mod array;
