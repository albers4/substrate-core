// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

mod access;
mod arr_macro;
mod core;
mod error;
mod ops;
mod utils;
mod view;
mod view_access;
mod ffi;

pub use core::Array;
pub use view::ArrayView;

pub use ffi::CppArray;