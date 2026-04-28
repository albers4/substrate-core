// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{
    ArrayAccess, ArrayLike, ArrayLikeMut, index::ToIndex, memory_order::MemoryOrder,
    number::Number, storage::Storage,
};

use crate::array::{
    ArrayView,
    error::ArrayError,
    utils::{compute_strides, traversal_iters, unravel_index},
};

pub struct Array<T: Number, S: Storage<Item = T>> {
    pub(crate) storage: S,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
    pub(crate) order: MemoryOrder,
}

pub type OwnedArray<T> = Array<T, Vec<T>>;

impl Array<f64, Vec<f64>> {
    pub fn empty() -> Self {
        Self {
            storage: vec![],
            shape: vec![0],
            strides: vec![1],
            offset: 0,
            order: Default::default(),
        }
    }

    pub fn empty_like(shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        Self {
            storage: Vec::with_capacity(total),
            shape: shape.to_vec(),
            strides: compute_strides(shape, Default::default()),
            offset: 0,
            order: Default::default(),
        }
    }

    pub fn from_scalar(scalar: f64) -> Self {
        Self {
            storage: vec![scalar],
            shape: vec![1],
            strides: vec![1],
            offset: 0,
            order: Default::default(),
        }
    }

    pub fn from_scalar_with_shape(scalar: f64, shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Self {
            storage: vec![scalar; size],
            shape: shape.to_vec(),
            strides: compute_strides(shape, Default::default()),
            offset: 0,
            order: Default::default(),
        }
    }

    pub fn from_vec_with_shape(data: Vec<f64>, shape: &[usize]) -> Result<Self, ArrayError> {
        if shape.is_empty() {
            return Err(ArrayError::EmptyShape);
        }
        if shape.contains(&0) {
            return Err(ArrayError::InvalidShapeDimension);
        }
        if data.len() != shape.iter().product::<usize>() {
            return Err(ArrayError::DimensionMismatch);
        }

        Ok(Self {
            storage: data,
            shape: shape.to_vec(),
            strides: compute_strides(shape, Default::default()),
            offset: 0,
            order: Default::default(),
        })
    }
}

impl ArrayLike for Array<f64, Vec<f64>> {
    type View<'a> = ArrayView<'a, f64>;

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

    fn reshape(self, new_shape: &[usize]) -> Result<Self, Self::Error> {
        if new_shape.is_empty() {
            return Err(ArrayError::EmptyShape);
        }
        if new_shape.contains(&0) {
            return Err(ArrayError::InvalidShapeDimension);
        }
        if new_shape.iter().product::<usize>() != self.length() {
            return Err(ArrayError::ReshapeSizeMismatch);
        }
        if !self.is_contiguous() {
            return Err(ArrayError::NotContiguous);
        }

        let strides = compute_strides(new_shape, self.order);
        Ok(Array {
            storage: self.storage,
            shape: new_shape.to_vec(),
            strides,
            offset: self.offset,
            order: self.order,
        })
    }

    fn into_shape(self, new_shape: &[usize]) -> Result<Self, Self::Error> {
        if new_shape.is_empty() {
            return Err(ArrayError::EmptyShape);
        }
        if new_shape.contains(&0) {
            return Err(ArrayError::InvalidShapeDimension);
        }
        if new_shape.iter().product::<usize>() != self.length() {
            return Err(ArrayError::ReshapeSizeMismatch);
        }

        let contiguous = if self.is_contiguous() {
            self
        } else {
            match self.order {
                MemoryOrder::RowMajor => self.to_row_major()?,
                MemoryOrder::ColumnMajor => self.to_column_major()?,
            }
        };

        contiguous.reshape(new_shape)
    }

    fn to_row_major(self) -> Result<Self, Self::Error> {
        if self.is_canonical(MemoryOrder::RowMajor) {
            return Ok(self);
        }

        let mut row_major_storage = vec![0.0f64; self.storage_length()];

        for (i, dst) in row_major_storage.iter_mut().enumerate() {
            let row_indices = unravel_index(i, self.shape(), MemoryOrder::RowMajor)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            let src_index: usize = self
                .physical_from_indices(&row_indices)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            *dst = *self.get_flat(src_index)?;
        }

        Ok(Array {
            storage: row_major_storage,
            shape: self.shape.to_vec(),
            strides: compute_strides(self.shape(), MemoryOrder::RowMajor),
            offset: 0,
            order: MemoryOrder::RowMajor,
        })
    }

    fn to_column_major(self) -> Result<Self, Self::Error> {
        if self.is_canonical(MemoryOrder::ColumnMajor) {
            return Ok(self);
        }

        let mut column_major_storage = vec![0.0f64; self.storage_length()];

        for (i, dst) in column_major_storage.iter_mut().enumerate() {
            let column_indices = unravel_index(i, self.shape(), MemoryOrder::ColumnMajor)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            let src_index: usize = self
                .physical_from_indices(&column_indices)
                .map_err(|_| ArrayError::IndexOutOfBounds)?;
            *dst = *self.get_flat(src_index)?;
        }

        Ok(Array {
            storage: column_major_storage,
            shape: self.shape.to_vec(),
            strides: compute_strides(self.shape(), MemoryOrder::ColumnMajor),
            offset: 0,
            order: MemoryOrder::ColumnMajor,
        })
    }

    fn view(&self) -> Self::View<'_> {
        ArrayView {
            data: &self.storage,
            shape: self.shape.to_vec(),
            strides: self.strides.to_vec(),
            offset: self.offset,
            order: self.order,
        }
    }
}

impl ArrayLikeMut for Array<f64, Vec<f64>> {
    fn set_flat(&mut self, index: impl ToIndex, value: Self::Item) -> Result<(), Self::Error> {
        let idx = index
            .to_index()
            .map_err(|_| ArrayError::IndexConversionError)?;
        if idx >= self.size() {
            return Err(ArrayError::IndexOutOfBounds);
        }

        let pairs = traversal_iters(self.shape.to_vec(), self.strides.to_vec(), self.order);
        let mut flat_index = self.offset();
        let mut temp = idx;
        for &(dim, stride) in &pairs {
            flat_index += (temp % dim) * stride;
            temp /= dim;
        }
        self.storage.as_mut_slice()[flat_index] = value;

        Ok(())
    }

    fn set(&mut self, indices: &[impl ToIndex], value: Self::Item) -> Result<(), Self::Error> {
        let index = self.physical_from_indices(indices)?;
        self.storage.as_mut_slice()[index] = value;
        Ok(())
    }

    unsafe fn set_unchecked(
        &mut self,
        indices: &[impl ToIndex],
        value: Self::Item,
    ) -> Result<(), Self::Error> {
        let mut index: usize = self.offset();
        for (i, idx) in indices.iter().enumerate() {
            let dim = idx.to_index().unwrap();
            index += dim * self.strides[i];
        }
        unsafe { *self.storage.as_mut_ptr().add(index) = value }
        Ok(())
    }
}
