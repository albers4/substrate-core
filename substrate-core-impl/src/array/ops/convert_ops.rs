// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{ArrayLike, ops::ConvertOps, ops::AccessOps};

use crate::{Array, array::error::ArrayError};

impl ConvertOps for Array<f64, Vec<f64>> {
    type Item = f64;
    type Error = ArrayError;

    fn to_scalar(&self) -> Result<Self::Item, Self::Error> {
        if self.length() != 1 {
            return Err(ArrayError::ArrayNotAScalar);
        }

        if let Ok(scalar) = self.get_flat(0) {
            Ok(*scalar)
        } else {
            Err(ArrayError::ArrayNotAScalar)
        }
    }

    fn to_vec(&self) -> Vec<Self::Item> {
        self.storage.as_slice().to_vec()
    }
}