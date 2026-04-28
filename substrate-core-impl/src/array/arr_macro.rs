// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#[macro_export]
macro_rules! arr {
    // ( )
    () => { $crate::Array::empty_like(&[0]) };
    // ( 0..=10 )
    ( $start:literal ..= $end:literal ) => {{
        let vec: Vec<_> = ($start..=$end).collect();
        let shape = vec![vec.len()];
        $crate::Array::from_vec_with_shape(vec, &shape).expect("Failed to create Array")
    }};
    // ( 0..10 )
    ( $start:literal .. $end:literal ) => {{
        let vec: Vec<_> = ($start..$end).collect();
        let shape = vec![vec.len()];
        $crate::Array::from_vec_with_shape(vec, &shape).expect("Failed to create Array")
    }};
    // ( 1, 2, 3 )
    ( $($l:literal),* $(,)? ) => { $crate::Array::from_vec(vec![$($l),*]) };
    // ( value )
    /*
    ( $value:expr ) => {
        $crate::Array::from_vec(vec![$value])
    };
    */
    // ( [...], [...], ... )
    ( $left:tt $(, $($right:tt),*),* $(,)? ) => {{
        let mut data = vec!();
        let left_array = $crate::__arr_nested!($left);
        let mut shape: Vec<usize> = left_array.shape();
        data.extend(left_array.data().to_vec());

        $(
            $(
                let next_array = $crate::__arr_nested!($right);
                assert_eq!(left_array.shape(), next_array.shape(), "Inconsistent shapes in array creation");
                data.extend(next_array.to_vec());
                shape[0] += 1;
            )*;
        )*;

        $crate::Array::from_vec_with_shape(data, &shape).expect("Failed to create Array")
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! __arr_nested {
    // ( [0..=10] )
    ( [$start:literal ..= $end:literal] ) => {{
        let mut shape = vec!(1);
        let array = $crate::arr!($start..=$end);
        shape.extend(array.shape());
        $crate::Array::from_vec_with_shape(array.to_vec(), &shape).expect("Failed to create Array")
    }};
    // ( [0..10] )
    ( [$start:literal .. $end:literal] ) => {{
        let mut shape = vec!(1);
        let array = $crate::arr!($start..$end);
        shape.extend(array.shape());
        $crate::Array::from_vec_with_shape(array.to_vec(), &shape).expect("Failed to create Array")
    }};
    // ( [ 1, 2, 3 ] )
    ( [$($l:literal),*] ) => {{
        let mut shape = vec!(1);
        let array = $crate::arr!($($l),*);
        shape.extend(array.shape());
        $crate::Array::from_vec_with_shape(array.to_vec(), &shape).expect("Failed to create Array")
    }};
    // ( [[...], [...], ...] )
    ( [$left:tt $(, $($right:tt),*),* $(,)?] ) => {{
        let mut data = vec!();
        let left_array = $crate::__arr_nested!($left);
        let mut shape: Vec<usize> = left_array.shape();
        data.extend(left_array.data().to_vec());

        $(
            $(
                let next_array = $crate::__arr_nested!($right);
                assert_eq!(left_array.shape(), next_array.shape(), "Inconsistent shapes in array creation");
                data.extend(next_array.to_vec());
                shape[0] += 1;
            )*;
        )*;

        shape.insert(0, 1); // Accounts for the brackets in the matching arm

        $crate::Array::from_vec_with_shape(data, &shape).expect("Failed to create Array")
    }};
}
