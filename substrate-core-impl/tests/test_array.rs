// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_impl::{Array, CppArray};
use substrate_core_spec::array::{ArrayLike, ops::InitOps};

#[test]
fn test_array_creation() {
    let arr = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(arr.length(), 5);
}

#[test]
fn test_array_op_add_forward_backward() {
    let a = CppArray::new(vec![0.0, 1.0, 3.0, 4.0, 5.0]);
    let b = CppArray::new(vec![0.0, 1.0, 3.0, 4.0, 5.0]);

    let da = CppArray::new(vec![1.0; 5]);
    let db = CppArray::new(vec![0.0; 5]);

    let dres_fwd = CppArray::add_forward(&a, &da, &b, &db);
    println!("dres_fwd={:#?}", dres_fwd.to_vec());

    let dres_seed = CppArray::new(vec![5.0, 5.0, 5.0, 5.0, 5.0]);

    let (grad_a, grad_b) = CppArray::add_backward(&a, &b, &dres_seed);
    println!("grad_a={:#?}", grad_a.to_vec());
    println!("grad_b={:#?}", grad_b.to_vec());
}
