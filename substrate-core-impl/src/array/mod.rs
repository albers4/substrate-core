// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

mod arr_macro;
mod core;
mod display;
mod error;
mod ffi;
mod ops;
mod utils;
mod view;

pub use core::Array;
pub use error::ArrayError;
pub use view::ArrayView;

pub use ffi::CppArray;
