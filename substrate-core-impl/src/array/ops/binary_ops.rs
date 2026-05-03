// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::ops::BinaryOps;

use crate::{Array, array::error::ArrayError};

impl BinaryOps for Array<f64, Vec<f64>> {
    type Output = Self;
    type Error = ArrayError;
    
    fn add(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn sub(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn mul(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn div(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn pow(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn rem(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn max(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn min(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

    
}