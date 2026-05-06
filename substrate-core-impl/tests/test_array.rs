// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_impl::{Array, ArrayError, CppArray};
use substrate_core_spec::array::{
    ArrayLike,
    ops::{AccessOps, ConvertOps, InitOps, LinearAlgebraOps, ShapeOps},
};

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

#[test]
fn test_squeeze() {
    // 1. Scalar (0‑dimensional) -> remains scalar (shape [])
    let a = Array::from_vec(vec![42.0]);
    let squeezed = a.view().squeeze().unwrap();
    assert_eq!(squeezed.shape(), &[]);
    assert_eq!(squeezed.to_scalar().unwrap(), 42.0);

    // 2. Remove single dimension of size 1 at beginning
    let a = Array::from_vec(vec![1.0, 2.0, 3.0])
        .reshape(&[1, 3])
        .unwrap();
    let squeezed = a.view().squeeze().unwrap();
    assert_eq!(squeezed.shape(), &[3]);
    assert_eq!(squeezed.to_vec(), vec![1.0, 2.0, 3.0]);

    // 3. Remove single dimension of size 1 at end
    let a = Array::from_vec(vec![1.0, 2.0, 3.0])
        .reshape(&[3, 1])
        .unwrap();
    let squeezed = a.view().squeeze().unwrap();
    assert_eq!(squeezed.shape(), &[3]);
    assert_eq!(squeezed.to_vec(), vec![1.0, 2.0, 3.0]);

    // 4. Remove multiple dimensions (shape [1,2,1,3,1] -> [2,3])
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .reshape(&[1, 2, 1, 3, 1])
        .unwrap();
    let squeezed = a.view().squeeze().unwrap();
    assert_eq!(squeezed.shape(), &[2, 3]);
    assert_eq!(squeezed.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // 5. Already no dimension of size 1 -> unchanged shape
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0])
        .reshape(&[2, 2])
        .unwrap();
    let squeezed = a.view().squeeze().unwrap();
    assert_eq!(squeezed.shape(), &[2, 2]);
    assert_eq!(squeezed.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

    // 6. Non‑contiguous view (transposed) – squeeze should still work
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .reshape(&[2, 3])
        .unwrap();
    let transposed = a.transpose().unwrap(); // shape [3,2], non‑contiguous
    let squeezed = transposed.squeeze().unwrap(); // still [3,2]
    assert_eq!(squeezed.shape(), &[3, 2]);
    assert_eq!(squeezed.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    // 7. Sliced view with singleton dimension after slicing
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .reshape(&[2, 3])
        .unwrap();
    let sliced = a.slice_by_range(0, 1..2).unwrap(); // shape [1,3]
    let squeezed = sliced.squeeze().unwrap();
    assert_eq!(squeezed.shape(), &[3]);
    assert_eq!(squeezed.to_vec(), vec![4.0, 5.0, 6.0]);

    // 8. Empty array – should return error
    let a = Array::from_vec(Vec::<f64>::new());
    let result = a.view().squeeze().unwrap();
    assert_eq!(result.shape(), &[0]);

    // 9. Perform a squeeze that results in a scalar (all dimensions removed)
    let a = Array::from_vec(vec![99.0]).reshape(&[1, 1, 1]).unwrap();
    let squeezed = a.view().squeeze().unwrap();
    assert_eq!(squeezed.shape(), &[]);
    assert_eq!(squeezed.to_scalar().unwrap(), 99.0);
}
