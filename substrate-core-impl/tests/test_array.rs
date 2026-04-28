// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_impl::Array;
use substrate_core_spec::array::{ArrayAccess, ArrayLike};

#[test]
fn test_array_creation() {
    let arr = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(arr.length(), 5);
}
