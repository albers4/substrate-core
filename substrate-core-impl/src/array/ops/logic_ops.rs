// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::ops::LogicOps;

use crate::{Array, array::error::ArrayError};

impl LogicOps for Array<f64, Vec<f64>> {
    type Output = Self;
    type Error = ArrayError;
    
    fn is_finite(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn is_inf(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn is_nan(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn allclose(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn eq(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn neq(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn gt(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn lt(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn ge(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn le(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        todo!()
    }

}