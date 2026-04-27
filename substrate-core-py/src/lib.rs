use pyo3::prelude::*;
use pyo3::types::PyModule;
use substrate_core_impl::DenseArray;

#[pyclass]
struct PyDenseArray {
    inner: DenseArray<f64>,
}

#[pymethods]
impl PyDenseArray {
    #[new]
    fn new(size: usize) -> Self {
        Self {
            inner: DenseArray::new(size),
        }
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
}

#[pymodule]
fn substrate_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDenseArray>()?;
    Ok(())
}
