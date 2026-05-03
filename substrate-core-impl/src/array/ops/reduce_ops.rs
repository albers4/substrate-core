// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::ops::ReduceOps;

use crate::{Array, array::error::ArrayError};

impl ReduceOps for Array<f64, Vec<f64>> {
    type Output = Self;
    type Error = ArrayError;
    
    fn sum(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn sum_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn mean(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn mean_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn var(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn var_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn std(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn std_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn prod(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn prod_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn min(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn min_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn max(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn max_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn argmin(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn argmin_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn argmax(&self) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn argmax_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn any(&self, value: impl substrate_core_spec::array::number::Number) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn any_axis(&self, value: impl substrate_core_spec::array::number::Number, axis: usize) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn all(&self, value: impl substrate_core_spec::array::number::Number) -> Result<Self::Output, Self::Error> {
        todo!()
    }
    
    fn all_axis(&self, value: impl substrate_core_spec::array::number::Number, axis: usize) -> Result<Self::Output, Self::Error> {
        todo!()
    }

}