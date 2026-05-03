// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

mod arr_macro;
mod core;
mod error;
mod utils;
mod view;
mod ffi;
mod ops;

pub use core::Array;
pub use view::ArrayView;

pub use ffi::CppArray;