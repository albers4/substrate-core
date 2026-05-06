// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use pyo3::prelude::*;
use pyo3::types::PyModule;
use substrate_core_impl::Array;
use substrate_core_spec::array::{ArrayLike, ops::InitOps};

#[pyclass]
struct PyArray {
    inner: Array<f64, Vec<f64>>,
}

#[pymethods]
impl PyArray {
    #[new]
    fn new(_size: usize) -> Self {
        Self {
            inner: Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]),
        }
    }

    pub fn length(&self) -> usize {
        self.inner.length()
    }
}

#[pymodule]
fn substrate_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArray>()?;
    Ok(())
}
