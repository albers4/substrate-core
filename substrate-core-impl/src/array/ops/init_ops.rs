// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::ops::InitOps;

use crate::{Array, array::error::ArrayError};

impl InitOps for Array<f64, Vec<f64>> {
    type Item = f64;
    type Output = Self;
    type Error = ArrayError;

    fn rand() {
        todo!()
    }

    fn randn() {
        todo!()
    }

    fn eye() {
        todo!()
    }

    fn diag() {
        todo!()
    }

    fn full() {
        todo!()
    }

    fn arange() {
        todo!()
    }

    fn logspace() {
        todo!()
    }

    fn zeros(shape: &[usize]) -> Self {
        Self::from_scalar_with_shape(0.0f64, shape)
    }

    fn ones(shape: &[usize]) -> Self {
        Self::from_scalar_with_shape(1.0f64, shape)
    }

    fn from_vec(vec: Vec<Self::Item>) -> Self {
        Self {
            shape: vec![vec.len()],
            storage: vec,
            strides: vec![1],
            offset: 0,
            order: Default::default(),
        }
    }

    fn linspace(a: Self::Item, b: Self::Item, n: usize) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        if n == 0 {
            return Err(ArrayError::ArrayFromLinspaceError);
        }

        let mut data = Vec::with_capacity(n);

        if n == 1 {
            data.push(a);
        } else {
            let n_minus_1 = (n - 1) as f64;
            let step = (b - a) / n_minus_1;
            let mut val = a;
            for _ in 0..n {
                data.push(val);
                val += step;
            }

            if n > 0 {
                *data.last_mut().unwrap() = b;
            }
        }

        Ok(Self {
            storage: data,
            shape: vec![n],
            strides: vec![1],
            offset: 0,
            order: Default::default(),
        })
    }

    fn from_fn<F>(shape: &[usize], mut f: F) -> Result<Self, Self::Error>
    where
        F: FnMut(&[usize]) -> Self::Item,
    {
        if shape.is_empty() {
            let storage = vec![f(&[])];
            return Ok(Self::from_vec(storage));
        }

        let size: usize = shape.iter().product();
        let mut storage: Vec<f64> = Vec::with_capacity(size);
        let mut indices: Vec<usize> = vec![0; shape.len()];

        fn generate<F>(
            dim: usize,
            indices: &mut [usize],
            shape: &[usize],
            storage: &mut Vec<f64>,
            f: &mut F,
        ) where
            F: FnMut(&[usize]) -> f64,
        {
            if dim == shape.len() {
                storage.push(f(indices));
                return;
            }
            for i in 0..shape[dim] {
                indices[dim] = i;
                generate(dim + 1, indices, shape, storage, f);
            }
        }
        generate(0, &mut indices, shape, &mut storage, &mut f);

        Array::from_vec_with_shape(storage, shape)
    }
}