// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{
    ArrayLike,
    ops::{AccessOps, BinaryOps, ConvertOps},
};

use crate::{
    Array,
    array::{
        ArrayView,
        error::ArrayError,
        utils::{broadcast_shapes, broadcast_strides},
    },
};

impl<'a> ArrayView<'a, f64> {
    fn add_same_shape<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let data = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a + b)
            .collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn add_scalar<Rhs: AccessOps<Item = f64, Error = ArrayError>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let scalar = *other.first()?;
        let data: Vec<f64> = self.data().iter().map(|a| a + scalar).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn add_broadcast<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let target_shape = broadcast_shapes(self.shape(), other.shape())?;
        let total_len = target_shape.iter().product();

        let strides_self = broadcast_strides(self.shape(), self.strides(), &target_shape)?;
        let strides_other = broadcast_strides(other.shape(), other.strides(), &target_shape)?;

        let offset_self = self.offset();
        let offset_other = other.offset();

        let mut data: Vec<f64> = vec![0.0; total_len];

        for (flat_idx, flat_item) in data.iter_mut().enumerate() {
            let mut rem = flat_idx;
            let mut idx_self = offset_self;
            let mut idx_other = offset_other;

            for dim in (0..target_shape.len()).rev() {
                let coord = rem % target_shape[dim];
                rem /= target_shape[dim];
                idx_self += coord * strides_self[dim];
                idx_other += coord * strides_other[dim];
            }

            // Safety: Using `unsafe` to avoid bounds checks - safe because indices are correct
            unsafe {
                let a = *self.data.as_ptr().add(idx_self);
                let b = *other.data().as_ptr().add(idx_other);
                *flat_item = a + b;
            }
        }

        Array::from_vec_with_shape(data, &target_shape)
    }

    fn sub_same_shape<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let data = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a - b)
            .collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn sub_scalar<Rhs: AccessOps<Item = f64, Error = ArrayError>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let scalar = *other.first()?;
        let data: Vec<f64> = self.data().iter().map(|a| a - scalar).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn sub_broadcast<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let target_shape = broadcast_shapes(self.shape(), other.shape())?;
        let total_len = target_shape.iter().product();

        let strides_self = broadcast_strides(self.shape(), self.strides(), &target_shape)?;
        let strides_other = broadcast_strides(other.shape(), other.strides(), &target_shape)?;

        let offset_self = self.offset();
        let offset_other = other.offset();

        let mut data: Vec<f64> = vec![0.0; total_len];

        for (flat_idx, flat_item) in data.iter_mut().enumerate() {
            let mut rem = flat_idx;
            let mut idx_self = offset_self;
            let mut idx_other = offset_other;

            for dim in (0..target_shape.len()).rev() {
                let coord = rem % target_shape[dim];
                rem /= target_shape[dim];
                idx_self += coord * strides_self[dim];
                idx_other += coord * strides_other[dim];
            }

            // Safety: Using `unsafe` to avoid bounds checks - safe because indices are correct
            unsafe {
                let a = *self.data.as_ptr().add(idx_self);
                let b = *other.data().as_ptr().add(idx_other);
                *flat_item = a - b;
            }
        }

        Array::from_vec_with_shape(data, &target_shape)
    }

    fn mul_same_shape<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let data = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a * b)
            .collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn mul_scalar<Rhs: AccessOps<Item = f64, Error = ArrayError>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let scalar = *other.first()?;
        let data: Vec<f64> = self.data().iter().map(|a| a * scalar).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn mul_broadcast<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let target_shape = broadcast_shapes(self.shape(), other.shape())?;
        let total_len = target_shape.iter().product();

        let strides_self = broadcast_strides(self.shape(), self.strides(), &target_shape)?;
        let strides_other = broadcast_strides(other.shape(), other.strides(), &target_shape)?;

        let offset_self = self.offset();
        let offset_other = other.offset();

        let mut data: Vec<f64> = vec![0.0; total_len];

        for (flat_idx, flat_item) in data.iter_mut().enumerate() {
            let mut rem = flat_idx;
            let mut idx_self = offset_self;
            let mut idx_other = offset_other;

            for dim in (0..target_shape.len()).rev() {
                let coord = rem % target_shape[dim];
                rem /= target_shape[dim];
                idx_self += coord * strides_self[dim];
                idx_other += coord * strides_other[dim];
            }

            // Safety: Using `unsafe` to avoid bounds checks - safe because indices are correct
            unsafe {
                let a = *self.data.as_ptr().add(idx_self);
                let b = *other.data().as_ptr().add(idx_other);
                *flat_item = a * b;
            }
        }

        Array::from_vec_with_shape(data, &target_shape)
    }

    fn div_same_shape<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let data = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a / b)
            .collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn div_scalar<Rhs: AccessOps<Item = f64, Error = ArrayError>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let scalar = *other.first()?;
        let data: Vec<f64> = self.data().iter().map(|a| a / scalar).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn div_broadcast<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let target_shape = broadcast_shapes(self.shape(), other.shape())?;
        let total_len = target_shape.iter().product();

        let strides_self = broadcast_strides(self.shape(), self.strides(), &target_shape)?;
        let strides_other = broadcast_strides(other.shape(), other.strides(), &target_shape)?;

        let offset_self = self.offset();
        let offset_other = other.offset();

        let mut data: Vec<f64> = vec![0.0; total_len];

        for (flat_idx, flat_item) in data.iter_mut().enumerate() {
            let mut rem = flat_idx;
            let mut idx_self = offset_self;
            let mut idx_other = offset_other;

            for dim in (0..target_shape.len()).rev() {
                let coord = rem % target_shape[dim];
                rem /= target_shape[dim];
                idx_self += coord * strides_self[dim];
                idx_other += coord * strides_other[dim];
            }

            // Safety: Using `unsafe` to avoid bounds checks - safe because indices are correct
            unsafe {
                let a = *self.data.as_ptr().add(idx_self);
                let b = *other.data().as_ptr().add(idx_other);
                *flat_item = a / b;
            }
        }

        Array::from_vec_with_shape(data, &target_shape)
    }

    fn pow_same_shape<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let data = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a.powf(*b))
            .collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn pow_scalar<Rhs: AccessOps<Item = f64, Error = ArrayError>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let scalar = *other.first()?;
        let data: Vec<f64> = self.data().iter().map(|a| a.powf(scalar)).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn pow_broadcast<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let target_shape = broadcast_shapes(self.shape(), other.shape())?;
        let total_len = target_shape.iter().product();

        let strides_self = broadcast_strides(self.shape(), self.strides(), &target_shape)?;
        let strides_other = broadcast_strides(other.shape(), other.strides(), &target_shape)?;

        let offset_self = self.offset();
        let offset_other = other.offset();

        let mut data: Vec<f64> = vec![0.0; total_len];

        for (flat_idx, flat_item) in data.iter_mut().enumerate() {
            let mut rem = flat_idx;
            let mut idx_self = offset_self;
            let mut idx_other = offset_other;

            for dim in (0..target_shape.len()).rev() {
                let coord = rem % target_shape[dim];
                rem /= target_shape[dim];
                idx_self += coord * strides_self[dim];
                idx_other += coord * strides_other[dim];
            }

            // Safety: Using `unsafe` to avoid bounds checks - safe because indices are correct
            unsafe {
                let a = *self.data.as_ptr().add(idx_self);
                let b = *other.data().as_ptr().add(idx_other);
                *flat_item = a.powf(b);
            }
        }

        Array::from_vec_with_shape(data, &target_shape)
    }

    fn rem_same_shape<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let data = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a % b)
            .collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn rem_scalar<Rhs: AccessOps<Item = f64, Error = ArrayError>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let scalar = *other.first()?;
        let data: Vec<f64> = self.data().iter().map(|a| a % scalar).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn rem_broadcast<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let target_shape = broadcast_shapes(self.shape(), other.shape())?;
        let total_len = target_shape.iter().product();

        let strides_self = broadcast_strides(self.shape(), self.strides(), &target_shape)?;
        let strides_other = broadcast_strides(other.shape(), other.strides(), &target_shape)?;

        let offset_self = self.offset();
        let offset_other = other.offset();

        let mut data: Vec<f64> = vec![0.0; total_len];

        for (flat_idx, flat_item) in data.iter_mut().enumerate() {
            let mut rem = flat_idx;
            let mut idx_self = offset_self;
            let mut idx_other = offset_other;

            for dim in (0..target_shape.len()).rev() {
                let coord = rem % target_shape[dim];
                rem /= target_shape[dim];
                idx_self += coord * strides_self[dim];
                idx_other += coord * strides_other[dim];
            }

            // Safety: Using `unsafe` to avoid bounds checks - safe because indices are correct
            unsafe {
                let a = *self.data.as_ptr().add(idx_self);
                let b = *other.data().as_ptr().add(idx_other);
                *flat_item = a % b;
            }
        }

        Array::from_vec_with_shape(data, &target_shape)
    }

    fn max_same_shape<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let data = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a.max(*b))
            .collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn max_scalar<Rhs: AccessOps<Item = f64, Error = ArrayError>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let scalar = *other.first()?;
        let data: Vec<f64> = self.data().iter().map(|a| a.max(scalar)).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn max_broadcast<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let target_shape = broadcast_shapes(self.shape(), other.shape())?;
        let total_len = target_shape.iter().product();

        let strides_self = broadcast_strides(self.shape(), self.strides(), &target_shape)?;
        let strides_other = broadcast_strides(other.shape(), other.strides(), &target_shape)?;

        let offset_self = self.offset();
        let offset_other = other.offset();

        let mut data: Vec<f64> = vec![f64::NEG_INFINITY; total_len];

        for (flat_idx, flat_item) in data.iter_mut().enumerate() {
            let mut rem = flat_idx;
            let mut idx_self = offset_self;
            let mut idx_other = offset_other;

            for dim in (0..target_shape.len()).rev() {
                let coord = rem % target_shape[dim];
                rem /= target_shape[dim];
                idx_self += coord * strides_self[dim];
                idx_other += coord * strides_other[dim];
            }

            // Safety: Using `unsafe` to avoid bounds checks - safe because indices are correct
            unsafe {
                let a = *self.data.as_ptr().add(idx_self);
                let b = *other.data().as_ptr().add(idx_other);
                *flat_item = a.max(b);
            }
        }

        Array::from_vec_with_shape(data, &target_shape)
    }

    fn min_same_shape<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let data = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(a, b)| a.min(*b))
            .collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn min_scalar<Rhs: AccessOps<Item = f64, Error = ArrayError>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let scalar = *other.first()?;
        let data: Vec<f64> = self.data().iter().map(|a| a.min(scalar)).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    fn min_broadcast<Rhs: AccessOps<Item = f64>>(
        &self,
        other: &Rhs,
    ) -> Result<Array<f64, Vec<f64>>, ArrayError> {
        let target_shape = broadcast_shapes(self.shape(), other.shape())?;
        let total_len = target_shape.iter().product();

        let strides_self = broadcast_strides(self.shape(), self.strides(), &target_shape)?;
        let strides_other = broadcast_strides(other.shape(), other.strides(), &target_shape)?;

        let offset_self = self.offset();
        let offset_other = other.offset();

        let mut data: Vec<f64> = vec![f64::INFINITY; total_len];

        for (flat_idx, flat_item) in data.iter_mut().enumerate() {
            let mut rem = flat_idx;
            let mut idx_self = offset_self;
            let mut idx_other = offset_other;

            for dim in (0..target_shape.len()).rev() {
                let coord = rem % target_shape[dim];
                rem /= target_shape[dim];
                idx_self += coord * strides_self[dim];
                idx_other += coord * strides_other[dim];
            }

            // Safety: Using `unsafe` to avoid bounds checks - safe because indices are correct
            unsafe {
                let a = *self.data.as_ptr().add(idx_self);
                let b = *other.data().as_ptr().add(idx_other);
                *flat_item = a.min(b);
            }
        }

        Array::from_vec_with_shape(data, &target_shape)
    }
}

impl<'a> BinaryOps for ArrayView<'a, f64> {
    type Output = Array<f64, Vec<f64>>;

    /// Adds two Arrays.
    ///
    /// # Returns
    /// `Array<f64, Vec<f64>>`
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{BinaryOps, InitOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
    /// let c = a.add(&b).unwrap(); // element-by-element
    /// assert_eq!(c.to_vec(), vec![5.0, 7.0, 9.0]);
    ///
    /// let scalar = Array::from_vec(vec![10.0]);
    /// let d = a.add(&scalar).unwrap(); // scalar
    /// assert_eq!(d.to_vec(), vec![11.0, 12.0, 13.0]);
    ///
    /// let e = Array::from_vec(vec![1.0, 2.0]);
    /// let f = Array::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2].as_slice()).unwrap();
    /// let g = e.add(&f).unwrap(); // broadcast
    /// assert_eq!(g.to_vec(), vec![2.0, 4.0, 4.0, 6.0]);
    /// ```
    fn add<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        if self.shape() == other.shape() {
            self.add_same_shape(other)
        } else if other.ndim() == 0 {
            self.add_scalar(other)
        } else {
            self.add_broadcast(other)
        }
    }

    /// Subtracts two Arrays.
    ///
    /// # Returns
    /// `Array<f64, Vec<f64>>`
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{BinaryOps, InitOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
    /// let c = a.sub(&b).unwrap(); // element-by-element
    /// assert_eq!(c.to_vec(), vec![-3.0, -3.0, -3.0]);
    ///
    /// let scalar = Array::from_vec(vec![10.0]);
    /// let d = a.sub(&scalar).unwrap(); // scalar
    /// assert_eq!(d.to_vec(), vec![-9.0, -8.0, -7.0]);
    ///
    /// let e = Array::from_vec(vec![1.0, 2.0]);
    /// let f = Array::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2].as_slice()).unwrap();
    /// let g = e.sub(&f).unwrap(); // broadcast
    /// assert_eq!(g.to_vec(), vec![0.0, 0.0, -2.0, -2.0]);
    /// ```
    fn sub<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        if self.shape() == other.shape() {
            self.sub_same_shape(other)
        } else if other.ndim() == 0 {
            self.sub_scalar(other)
        } else {
            self.sub_broadcast(other)
        }
    }

    /// Multiply two Arrays.
    ///
    /// # Returns
    /// `Array<f64, Vec<f64>>`
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{BinaryOps, InitOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
    /// let c = a.mul(&b).unwrap(); // element-by-element
    /// assert_eq!(c.to_vec(), vec![4.0, 10.0, 18.0]);
    ///
    /// let scalar = Array::from_vec(vec![10.0]);
    /// let d = a.mul(&scalar).unwrap(); // scalar
    /// assert_eq!(d.to_vec(), vec![10.0, 20.0, 30.0]);
    ///
    /// let e = Array::from_vec(vec![1.0, 2.0]);
    /// let f = Array::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2].as_slice()).unwrap();
    /// let g = e.mul(&f).unwrap(); // broadcast
    /// assert_eq!(g.to_vec(), vec![1.0, 4.0, 3.0, 8.0]);
    /// ```
    fn mul<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        if self.shape() == other.shape() {
            self.mul_same_shape(other)
        } else if other.ndim() == 0 {
            self.mul_scalar(other)
        } else {
            self.mul_broadcast(other)
        }
    }

    /// Divide two Arrays.
    ///
    /// # Returns
    /// `Array<f64, Vec<f64>>`
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{BinaryOps, InitOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
    /// let c = a.div(&b).unwrap(); // element-by-element
    /// assert_eq!(c.to_vec(), vec![1.0/4.0, 2.0/5.0, 3.0/6.0]);
    ///
    /// let scalar = Array::from_vec(vec![10.0]);
    /// let d = a.div(&scalar).unwrap(); // scalar
    /// assert_eq!(d.to_vec(), vec![1.0/10.0, 2.0/10.0, 3.0/10.0]);
    ///
    /// let e = Array::from_vec(vec![1.0, 2.0]);
    /// let f = Array::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2].as_slice()).unwrap();
    /// let g = e.div(&f).unwrap(); // broadcast
    /// assert_eq!(g.to_vec(), vec![1.0/1.0, 2.0/2.0, 1.0/3.0, 2.0/4.0]);
    /// ```
    fn div<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        if self.shape() == other.shape() {
            self.div_same_shape(other)
        } else if other.ndim() == 0 {
            self.div_scalar(other)
        } else {
            self.div_broadcast(other)
        }
    }

    /// Power two Arrays.
    ///
    /// # Returns
    /// `Array<f64, Vec<f64>>`
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{BinaryOps, InitOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
    /// let c = a.pow(&b).unwrap(); // element-by-element
    /// assert_eq!(c.to_vec(), vec![1.0_f64.powf(4.0), 2.0_f64.powf(5.0), 3.0_f64.powf(6.0)]);
    ///
    /// let scalar = Array::from_vec(vec![10.0]);
    /// let d = a.pow(&scalar).unwrap(); // scalar
    /// assert_eq!(d.to_vec(), vec![1.0_f64.powf(10.0), 2.0_f64.powf(10.0), 3.0_f64.powf(10.0)]);
    ///
    /// let e = Array::from_vec(vec![1.0, 2.0]);
    /// let f = Array::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2].as_slice()).unwrap();
    /// let g = e.pow(&f).unwrap(); // broadcast
    /// assert_eq!(g.to_vec(), vec![1.0_f64.powf(1.0), 2.0_f64.powf(2.0), 1.0_f64.powf(3.0), 2.0_f64.powf(4.0)]);
    /// ```
    fn pow<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        if self.shape() == other.shape() {
            self.pow_same_shape(other)
        } else if other.ndim() == 0 {
            self.pow_scalar(other)
        } else {
            self.pow_broadcast(other)
        }
    }

    /// Remainder of two Arrays.
    ///
    /// # Returns
    /// `Array<f64, Vec<f64>>`
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{BinaryOps, InitOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
    /// let c = a.rem(&b).unwrap(); // element-by-element
    /// assert_eq!(c.to_vec(), vec![1.0 % 4.0, 2.0 % 5.0, 3.0 % 6.0]);
    ///
    /// let scalar = Array::from_vec(vec![2.0]);
    /// let d = a.rem(&scalar).unwrap(); // scalar
    /// assert_eq!(d.to_vec(), vec![1.0 % 2.0, 2.0 % 2.0, 3.0 % 2.0]);
    ///
    /// let e = Array::from_vec(vec![1.0, 2.0]);
    /// let f = Array::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2].as_slice()).unwrap();
    /// let g = e.rem(&f).unwrap(); // broadcast
    /// assert_eq!(g.to_vec(), vec![1.0 % 1.0, 2.0 % 2.0, 1.0 % 3.0, 2.0 % 4.0]);
    /// ```
    fn rem<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        if self.shape() == other.shape() {
            self.rem_same_shape(other)
        } else if other.ndim() == 0 {
            self.rem_scalar(other)
        } else {
            self.rem_broadcast(other)
        }
    }

    /// Maximum of two Arrays.
    ///
    /// # Returns
    /// `Array<f64, Vec<f64>>`
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{BinaryOps, InitOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
    /// let c = a.max(&b).unwrap(); // element-by-element
    /// assert_eq!(c.to_vec(), vec![4.0, 5.0, 6.0]);
    ///
    /// let scalar = Array::from_vec(vec![2.0]);
    /// let d = a.max(&scalar).unwrap(); // scalar
    /// assert_eq!(d.to_vec(), vec![2.0, 2.0, 3.0]);
    ///
    /// let e = Array::from_vec(vec![1.0, 2.0]);
    /// let f = Array::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2].as_slice()).unwrap();
    /// let g = e.max(&f).unwrap(); // broadcast
    /// assert_eq!(g.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    fn max<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        if self.shape() == other.shape() {
            self.max_same_shape(other)
        } else if other.ndim() == 0 {
            self.max_scalar(other)
        } else {
            self.max_broadcast(other)
        }
    }

    /// Minimum of two Arrays.
    ///
    /// # Returns
    /// `Array<f64, Vec<f64>>`
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{BinaryOps, InitOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
    /// let c = a.min(&b).unwrap(); // element-by-element
    /// assert_eq!(c.to_vec(), vec![1.0, 2.0, 3.0]);
    ///
    /// let scalar = Array::from_vec(vec![2.0]);
    /// let d = a.min(&scalar).unwrap(); // scalar
    /// assert_eq!(d.to_vec(), vec![1.0, 2.0, 2.0]);
    ///
    /// let e = Array::from_vec(vec![1.0, 2.0]);
    /// let f = Array::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2].as_slice()).unwrap();
    /// let g = e.min(&f).unwrap(); // broadcast
    /// assert_eq!(g.to_vec(), vec![1.0, 2.0, 1.0, 2.0]);
    /// ```
    fn min<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        if self.shape() == other.shape() {
            self.min_same_shape(other)
        } else if other.ndim() == 0 {
            self.min_scalar(other)
        } else {
            self.min_broadcast(other)
        }
    }
}

impl BinaryOps for Array<f64, Vec<f64>> {
    type Output = Array<f64, Vec<f64>>;

    fn add<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        self.view().add(other)
    }

    fn sub<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        self.view().sub(other)
    }

    fn mul<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        self.view().mul(other)
    }

    fn div<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        self.view().div(other)
    }

    fn pow<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        self.view().pow(other)
    }

    fn rem<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        self.view().rem(other)
    }

    fn max<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        self.view().max(other)
    }

    fn min<Rhs: AccessOps<Item = Self::Item, Error = Self::Error>>(
        &self,
        other: &Rhs,
    ) -> Result<Self::Output, Self::Error> {
        self.view().min(other)
    }
}
