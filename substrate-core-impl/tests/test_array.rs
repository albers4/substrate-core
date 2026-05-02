// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_impl::{Array, CppArray};
use substrate_core_spec::array::{ArrayAccess, ArrayLike};

#[test]
fn test_array_creation() {
    let arr = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(arr.length(), 5);
}

#[test]
fn test_array_op_add_grad() {
    let a = CppArray::new(vec![0.0, 1.0, 3.0, 4.0, 5.0]);
    let b = CppArray::new(vec![0.0, 1.0, 3.0, 4.0, 5.0]);
    let res = a.add(&b).unwrap();
    let res_vec = res.to_vec();

    let (da, db) = a.grad_add(&b).unwrap();
    let da_vec = da.to_vec();
    let db_vec = db.to_vec();

    println!("res={:#?}", res_vec);
    println!("da={:#?}", da_vec);
    println!("db={:#?}", db_vec);
}
