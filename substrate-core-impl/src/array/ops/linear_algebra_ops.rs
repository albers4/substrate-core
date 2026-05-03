// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{ops::LinearAlgebraOps, ArrayLike};

use crate::{Array, array::{ArrayView, error::ArrayError}};

impl LinearAlgebraOps for Array<f64, Vec<f64>> {
    type Output = Self;
    type Error = ArrayError;
    type View<'a> = ArrayView<'a, f64>;
    
    fn dot(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn matmul(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn transpose(&self) -> Result<Self::View<'_>, Self::Error> {
        if self.ndim() != 2 {
            return Err(ArrayError::ValidForMatricesOnly);
        }

        let new_shape = vec![self.shape[1], self.shape[0]];
        let new_strides = vec![self.strides[1], self.strides[0]];

        Ok(ArrayView {
            data: &self.storage,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            order: self.order,
        })
    }

    fn transpose_copy(&self) -> Result<Self, Self::Error> {
        if self.ndim() != 2 {
            return Err(ArrayError::ValidForMatricesOnly);
        }

        let (d0, d1) = (self.shape[0], self.shape[1]);
        let new_shape = vec![d1, d0];
        let new_strides = vec![self.strides[1], self.strides[0]];
        let mut new_data = vec![0.0; self.length()];

        for i in 0..d0 {
            for j in 0..d1 {
                let src_idx = self.offset() + i * self.strides[0] + j * self.strides[1];
                let dst_idx = j * new_strides[0] + i * new_strides[1];
                new_data[dst_idx] = self.storage.as_slice()[src_idx];
            }
        }

        Ok(Array {
            storage: new_data,
            shape: new_shape,
            strides: new_strides,
            offset: 0,
            order: self.order,
        })
    }

    
    fn trace(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn det(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn inv(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn solve(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn eig(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn svd(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn qr(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn cholesky(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn norm(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn cross(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

}