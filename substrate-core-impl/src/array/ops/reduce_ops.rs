// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::array::{
    ArrayLike,
    ops::{AccessOps, ConvertOps, ReduceOps},
};

use crate::{
    Array,
    array::{
        ArrayView,
        error::ArrayError,
        utils::{compute_strides, unravel_index},
    },
};

impl<'a> ReduceOps for ArrayView<'a, f64> {
    type Output = Array<f64, Vec<f64>>;

    /// Returns the sum of all elements in the array as a 0‑dimensional array.
    ///
    /// # Returns
    /// A scalar array (shape `[1]`) containing the sum.
    ///
    /// # Errors
    /// `ArrayError::EmptyArray` if the array is empty.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ReduceOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let v = a.view();
    /// let s = v.sum().unwrap();
    /// assert_eq!(s.to_scalar().unwrap(), 6.0);
    /// assert_eq!(s.shape(), [1]);
    /// ```
    fn sum(&self) -> Result<Self::Output, Self::Error> {
        if self.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let total = self.iter().fold(0.0, |acc, x| acc + x);
        Ok(Array::from_scalar(total))
    }

    /// Computes the sum of elements along the specified axis.
    ///
    /// The output array has one fewer dimension than the input; the axis given is removed.
    /// The function iterates over all elements of the view, groups them according to
    /// the reduced axis, and accumulates the sum into the corresponding position in
    /// the output array.
    ///
    /// # Arguments
    /// * `axis` – The axis along which to sum (0‑based).
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `axis >= ndim()`.
    /// * `ArrayError::InvalidShape` – if the output shape calculation fails (should not happen).
    /// * `ArrayError::IndexOutOfBounds` – if `get_flat` fails (should not happen for valid indices).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` containing the sum along the given axis.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ReduceOps, ShapeOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    /// use substrate_core_spec::array::memory_order::MemoryOrder;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .reshape_copy(&[2, 3])
    ///     .unwrap()
    ///     .to_column_major().unwrap();
    /// let view = a.view();
    /// let sum_axis_0 = view.sum_axis(0).unwrap(); // sum over rows -> shape [3]
    /// assert_eq!(sum_axis_0.to_vec(), vec![5.0, 7.0, 9.0]);
    /// assert_eq!(sum_axis_0.order(), MemoryOrder::ColumnMajor);
    /// assert_eq!(sum_axis_0.shape(), vec![3]);
    ///
    /// let sum_axis_1 = view.sum_axis(1).unwrap(); // sum over columns -> shape [2]
    /// assert_eq!(sum_axis_1.to_vec(), vec![6.0, 15.0]);
    /// ```
    fn sum_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }

        let out_shape = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .collect::<Vec<usize>>();
        let out_len = out_shape.iter().product::<usize>();
        let mut out_data = vec![0.0; out_len];
        let out_strides = compute_strides(&out_shape, self.order());

        for in_flat in 0..self.length() {
            let in_coords = unravel_index(in_flat, &self.shape, self.order)?;
            let out_coords = in_coords
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &c)| c)
                .collect::<Vec<usize>>();
            let out_flat = out_coords
                .iter()
                .enumerate()
                .fold(0, |idx, (i, &c)| idx + c * out_strides[i]);

            let val = self.get_flat(in_flat)?;
            out_data[out_flat] += *val;
        }

        Ok(Array {
            storage: out_data,
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            order: self.order,
        })
    }

    /// Returns the arithmetic mean of all elements as a 0‑dimensional array.
    ///
    /// # Errors
    /// `ArrayError::EmptyArray` if the array is empty.
    ///
    /// # Examples
    /// ```
    /// # use substrate_core_impl::Array;
    /// # use substrate_core_spec::array::ops::{InitOps, ReduceOps, ShapeOps, ConvertOps};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    /// let v = a.view();
    /// let m = v.mean().unwrap();
    /// assert_eq!(m.to_scalar().unwrap(), 2.5);
    /// ```
    fn mean(&self) -> Result<Self::Output, Self::Error> {
        if self.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let total = self.sum()?.to_scalar()?;
        let mean_val = total / self.length() as f64;
        Ok(Array::from_scalar(mean_val))
    }

    /// Computes the arithmetic mean (average) of elements along the specified axis.
    ///
    /// The output array has one fewer dimension than the input; the axis given is removed.
    /// The function sums all elements along the axis for each corresponding output position,
    /// then divides by the length of that axis to obtain the mean.
    ///
    /// # Arguments
    /// * `axis` – The axis along which to compute the mean (0‑based).
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `axis >= ndim()`.
    /// * `ArrayError::InvalidShape` – if output shape calculation fails (should not happen).
    /// * `ArrayError::IndexOutOfBounds` – if `get_flat` fails (should not happen for valid indices).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` containing the mean along the given axis.
    /// The memory order (`RowMajor` or `ColumnMajor`) of the output is the same as the input.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ReduceOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .reshape_copy(&[2, 3])
    ///     .unwrap();
    /// let view = a.view();
    /// let mean_axis_0 = view.mean_axis(0).unwrap(); // mean over rows → shape [3]
    /// assert_eq!(mean_axis_0.to_vec(), vec![2.5, 3.5, 4.5]); // column‑wise means
    ///
    /// let mean_axis_1 = view.mean_axis(1).unwrap(); // mean over columns → shape [2]
    /// assert_eq!(mean_axis_1.to_vec(), vec![2.0, 5.0]); // row‑wise means
    /// ```
    fn mean_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }

        let out_shape = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .collect::<Vec<usize>>();
        let out_len = out_shape.iter().product::<usize>();
        let mut out_data = vec![0.0; out_len];
        let out_strides = compute_strides(&out_shape, self.order());
        let axis_len = self.shape[axis] as f64;

        for in_flat in 0..self.length() {
            let in_coords = unravel_index(in_flat, &self.shape, self.order)?;
            let out_coords = in_coords
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &c)| c)
                .collect::<Vec<usize>>();
            let out_flat = out_coords
                .iter()
                .enumerate()
                .fold(0, |idx, (i, &c)| idx + c * out_strides[i]);

            let val = self.get_flat(in_flat)?;
            out_data[out_flat] += *val;
        }

        for val in &mut out_data {
            *val /= axis_len
        }

        Ok(Array {
            storage: out_data,
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            order: self.order,
        })
    }

    /// Returns the sample variance of all elements (unbiased) as a 0‑dimensional array.
    ///
    /// Variance is computed as `SUM (x - mean)^2 / (n-1)`.
    ///
    /// # Errors
    /// `ArrayError::EmptyArray` if the array has fewer than 2 elements.
    ///
    /// # Examples
    /// ```
    /// # use substrate_core_impl::Array;
    /// # use substrate_core_spec::array::ops::{InitOps, ReduceOps, ConvertOps};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    /// let v = a.view();
    /// let var = v.var().unwrap();
    /// assert!((var.to_scalar().unwrap() - 1.6667).abs() < 1e-4);
    /// ```
    fn var(&self) -> Result<Self::Output, Self::Error> {
        if self.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let n = self.length() as f64;
        let (sum, sum_sq) = self
            .iter()
            .fold((0.0, 0.0), |(s, ss), x| (s + x, ss + x * x));
        let mean = sum / n;
        // Sample variance
        let variance = (sum_sq - n * mean * mean) / (n - 1.0);
        Ok(Array::from_scalar(variance))
    }

    /// Computes the sample variance of elements along the specified axis.
    ///
    /// The output array has one fewer dimension than the input. Variance is calculated as
    /// `(SUM (x_i - mean)^2) / (n - 1)` where `n` is the length of the axis.
    ///
    /// # Arguments
    /// * `axis` – The axis along which to compute variance (0‑based).
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `axis >= ndim()`.
    /// * `ArrayError::InvalidAxisLength` – if `self.shape[axis] <= 1` (cannot compute sample variance).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` containing the variance along the given axis.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ReduceOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .reshape_copy(&[2, 3]).unwrap();
    /// let view = a.view();
    /// let var_axis_0 = view.var_axis(0).unwrap(); // shape [3]
    /// assert_eq!(var_axis_0.to_vec(), vec![4.5, 4.5, 4.5]); // variance of each column
    /// ```
    fn var_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }

        let n = self.shape[axis] as f64;
        if n <= 1.0 {
            return Err(ArrayError::InvalidAxisLength);
        }

        let out_shape = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .collect::<Vec<usize>>();
        let out_len = out_shape.iter().product::<usize>();
        let mut out_data: Vec<f64> = vec![0.0; out_len];
        let out_strides = compute_strides(&out_shape, self.order());

        let mut sums = vec![0.0; out_len];
        for in_flat in 0..self.length() {
            let in_coords = unravel_index(in_flat, &self.shape, self.order)?;
            let out_coords = in_coords
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &c)| c)
                .collect::<Vec<usize>>();
            let out_flat = out_coords
                .iter()
                .enumerate()
                .fold(0, |idx, (i, &c)| idx + c * out_strides[i]);

            let val = self.get_flat(in_flat)?;
            sums[out_flat] += *val;
        }

        for in_flat in 0..self.length() {
            let in_coords = unravel_index(in_flat, &self.shape, self.order)?;
            let out_coords = in_coords
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &c)| c)
                .collect::<Vec<usize>>();
            let out_flat = out_coords
                .iter()
                .enumerate()
                .fold(0, |idx, (i, &c)| idx + c * out_strides[i]);

            let val = self.get_flat(in_flat)?;
            let mean = sums[out_flat] / n;
            let diff = *val - mean;
            out_data[out_flat] += diff * diff;
        }
        for v in &mut out_data {
            *v /= n - 1.0;
        }

        Ok(Array {
            storage: out_data,
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            order: self.order,
        })
    }

    /// Returns the sample standard deviation of all elements as a 0‑dimensional array.
    ///
    /// `std = sqrt(var)`.
    ///
    /// # Errors
    /// Same as `var`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ReduceOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    /// let v = a.view();
    /// let s = v.std().unwrap();
    /// assert!((s.to_scalar().unwrap() - 1.29099).abs() < 1e-4);
    /// ```
    fn std(&self) -> Result<Self::Output, Self::Error> {
        let variance = self.var()?.to_scalar()?;
        Ok(Array::from_scalar(variance.sqrt()))
    }

    /// Computes the sample standard deviation of elements along the specified axis.
    ///
    /// The output array has one fewer dimension than the input. Standard deviation is the
    /// square root of the variance (unbiased sample variance).
    ///
    /// # Arguments
    /// * `axis` – The axis along which to compute standard deviation (0‑based).
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `axis >= ndim()`.
    /// * `ArrayError::InvalidAxisLength` – if `self.shape[axis] <= 1`.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` containing the standard deviation along the given axis.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ReduceOps, ShapeOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .reshape_copy(&[2, 3]).unwrap();
    /// let view = a.view();
    /// let std_axis_0 = view.std_axis(0).unwrap();
    /// assert!(std_axis_0.to_vec().iter().all(|&x| (x - 2.1213).abs() < 1e-4));
    /// ```
    fn std_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        let var = self.var_axis(axis)?;
        let mut out_data = var.storage;
        for v in &mut out_data {
            *v = v.sqrt();
        }

        Ok(Array {
            storage: out_data,
            shape: var.shape,
            strides: var.strides,
            offset: 0,
            order: self.order,
        })
    }

    /// Returns the product of all elements as a 0‑dimensional array.
    ///
    /// # Errors
    /// `ArrayError::EmptyArray` if the array is empty.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ReduceOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    /// let v = a.view();
    /// let p = v.prod().unwrap();
    /// assert_eq!(p.to_scalar().unwrap(), 24.0);
    /// ```
    fn prod(&self) -> Result<Self::Output, Self::Error> {
        if self.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let prod = self.iter().fold(1.0, |acc, x| acc * x);
        Ok(Array::from_scalar(prod))
    }

    /// Computes the product of elements along the specified axis.
    ///
    /// The output array has one fewer dimension than the input; the axis given is removed.
    /// The function iterates over all elements of the view, groups them according to
    /// the reduced axis, and accumulates the product into the corresponding position in
    /// the output array (initialised with `1.0` before multiplication).
    ///
    /// # Arguments
    /// * `axis` – The axis along which to compute the product (0‑based).
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `axis >= ndim()`.
    /// * `ArrayError::EmptyArray` – if the array is empty (optional, but product of empty is undefined).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` containing the product along the given axis.
    /// The memory order (`RowMajor` or `ColumnMajor`) of the output is the same as the input.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps, ReduceOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ///     .reshape_copy(&[2, 3])
    ///     .unwrap();
    /// let v = a.view();
    /// let prod_axis_0 = v.prod_axis(0).unwrap(); // product over rows -> shape [3]
    /// assert_eq!(prod_axis_0.to_vec(), vec![4.0, 10.0, 18.0]); // column‑wise products
    ///
    /// let prod_axis_1 = v.prod_axis(1).unwrap(); // product over columns -> shape [2]
    /// assert_eq!(prod_axis_1.to_vec(), vec![6.0, 120.0]); // row‑wise products
    /// ```
    fn prod_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }

        let out_shape = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .collect::<Vec<usize>>();
        let out_len = out_shape.iter().product::<usize>();
        let mut out_data: Vec<f64> = vec![1.0; out_len];
        let out_strides = compute_strides(&out_shape, self.order());

        for in_flat in 0..self.length() {
            let in_coords = unravel_index(in_flat, &self.shape, self.order)?;
            let out_coords = in_coords
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &c)| c)
                .collect::<Vec<usize>>();
            let out_flat = out_coords
                .iter()
                .enumerate()
                .fold(0, |idx, (i, &c)| idx + c * out_strides[i]);

            let val = self.get_flat(in_flat)?;
            out_data[out_flat] *= *val;
        }

        Ok(Array {
            storage: out_data,
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            order: self.order,
        })
    }

    /// Returns the minimum value of all elements as a 0‑dimensional array.
    ///
    /// # Errors
    /// `ArrayError::EmptyArray` if the array is empty.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ReduceOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![3.0, -1.0, 4.0]);
    /// let v = a.view();
    /// let m = a.min().unwrap();
    /// assert_eq!(m.to_scalar().unwrap(), -1.0);
    /// ```
    fn min(&self) -> Result<Self::Output, Self::Error> {
        if self.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let min_val = self.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        Ok(Array::from_scalar(min_val))
    }

    /// Computes the minimum of elements along the specified axis.
    ///
    /// The output array has one fewer dimension than the input; the axis given is removed.
    /// The function iterates over all elements of the view, groups them according to
    /// the reduced axis, and accumulates the element‑wise minimum into the corresponding
    /// position in the output array.
    ///
    /// # Arguments
    /// * `axis` – The axis along which to compute the minimum (0‑based).
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `axis >= ndim()`.
    /// * `ArrayError::InvalidShape` – if output shape calculation fails (should not happen).
    /// * `ArrayError::IndexOutOfBounds` – if `get_flat` fails (should not happen for valid indices).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` containing the minimum along the given axis.
    /// The memory order (`RowMajor` or `ColumnMajor`) of the output is the same as the input.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps, ReduceOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 3.0, 2.0, 4.0, 0.0, 5.0])
    ///     .reshape_copy(&[2, 3])
    ///     .unwrap();
    /// let view = a.view();
    /// let min_axis_0 = view.min_axis(0).unwrap(); // min over rows -> shape [3]
    /// assert_eq!(min_axis_0.to_vec(), vec![1.0, 0.0, 2.0]); // column‑wise minima
    ///
    /// let min_axis_1 = view.min_axis(1).unwrap(); // min over columns -> shape [2]
    /// assert_eq!(min_axis_1.to_vec(), vec![1.0, 0.0]); // row‑wise minima
    /// ```
    fn min_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }

        let out_shape = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .collect::<Vec<usize>>();
        let out_len = out_shape.iter().product::<usize>();
        let mut out_data: Vec<f64> = vec![f64::INFINITY; out_len];
        let out_strides = compute_strides(&out_shape, self.order());

        for in_flat in 0..self.length() {
            let in_coords = unravel_index(in_flat, &self.shape, self.order)?;
            let out_coords = in_coords
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &c)| c)
                .collect::<Vec<usize>>();
            let out_flat = out_coords
                .iter()
                .enumerate()
                .fold(0, |idx, (i, &c)| idx + c * out_strides[i]);

            let val = self.get_flat(in_flat)?;
            out_data[out_flat] = out_data[out_flat].min(*val);
        }

        Ok(Array {
            storage: out_data,
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            order: self.order,
        })
    }

    /// Returns the maximum value of all elements as a 0‑dimensional array.
    ///
    /// # Errors
    /// `ArrayError::EmptyArray` if the array is empty.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ReduceOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![3.0, -1.0, 4.0]);
    /// let v = a.view();
    /// let m = a.max().unwrap();
    /// assert_eq!(m.to_scalar().unwrap(), 4.0);
    /// ```
    fn max(&self) -> Result<Self::Output, Self::Error> {
        if self.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let max_val = self.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        Ok(Array::from_scalar(max_val))
    }

    /// Computes the maximum of elements along the specified axis.
    ///
    /// The output array has one fewer dimension than the input; the axis given is removed.
    /// The function iterates over all elements of the view, groups them according to
    /// the reduced axis, and accumulates the element‑wise maximum into the corresponding
    /// position in the output array.
    ///
    /// # Arguments
    /// * `axis` – The axis along which to compute the maximum (0‑based).
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `axis >= ndim()`.
    /// * `ArrayError::InvalidShape` – if output shape calculation fails (should not happen).
    /// * `ArrayError::IndexOutOfBounds` – if `get_flat` fails (should not happen for valid indices).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` containing the minimum along the given axis.
    /// The memory order (`RowMajor` or `ColumnMajor`) of the output is the same as the input.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ConvertOps, ReduceOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 3.0, 2.0, 4.0, 0.0, 5.0])
    ///     .reshape_copy(&[2, 3])
    ///     .unwrap();
    /// let view = a.view();
    /// let max_axis_0 = view.max_axis(0).unwrap(); // max over rows -> shape [3]
    /// assert_eq!(max_axis_0.to_vec(), vec![4.0, 3.0, 5.0]); // column‑wise maxima
    ///
    /// let max_axis_1 = view.max_axis(1).unwrap(); // max over columns -> shape [2]
    /// assert_eq!(max_axis_1.to_vec(), vec![3.0, 5.0]); // row‑wise maxima
    /// ```
    fn max_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }

        let out_shape = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .collect::<Vec<usize>>();
        let out_len = out_shape.iter().product::<usize>();
        let mut out_data: Vec<f64> = vec![f64::NEG_INFINITY; out_len];
        let out_strides = compute_strides(&out_shape, self.order());

        for in_flat in 0..self.length() {
            let in_coords = unravel_index(in_flat, &self.shape, self.order)?;
            let out_coords = in_coords
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &c)| c)
                .collect::<Vec<usize>>();
            let out_flat = out_coords
                .iter()
                .enumerate()
                .fold(0, |idx, (i, &c)| idx + c * out_strides[i]);

            let val = self.get_flat(in_flat)?;
            out_data[out_flat] = out_data[out_flat].max(*val);
        }

        Ok(Array {
            storage: out_data,
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            order: self.order,
        })
    }

    /// Returns the flat index of the first occurrence of the minimum value.
    ///
    /// The index is returned as a 0‑dimensional array containing a `f64` (index as float).
    ///
    /// # Errors
    /// `ArrayError::EmptyArray` if the array is empty.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ReduceOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![3.0, -1.0, 4.0, -1.0]);
    /// let v = a.view();
    /// let idx = v.argmin().unwrap();
    /// assert_eq!(idx.to_scalar().unwrap(), 1.0); // first occurrence at index 1
    /// ```
    fn argmin(&self) -> Result<Self::Output, Self::Error> {
        if self.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let (idx, _) =
            self.iter()
                .enumerate()
                .fold((0, f64::INFINITY), |(best_idx, best_val), (i, &x)| {
                    if x < best_val {
                        (i, x)
                    } else {
                        (best_idx, best_val)
                    }
                });
        Ok(Array::from_scalar(idx as f64))
    }

    /// Returns the indices (flattened along the reduced axis) of the minimum values along the given axis.
    ///
    /// The output array has the same shape as the input with the specified axis removed. Each entry
    /// contains the linear index (within the reduced axis) of the first occurrence of the minimum
    /// value along that axis.
    ///
    /// # Arguments
    /// * `axis` – The axis along which to find the index of the minimum (0‑based).
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `axis >= ndim()`.
    /// * `ArrayError::EmptyArray` – if the input is empty.
    ///
    /// # Returns
    /// An `Array<f64, Vec<f64>>` where each element is the index (as `f64`) of the minimum
    /// along the reduced axis.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ReduceOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 5.0, 2.0, 3.0, 0.0, 4.0])
    ///     .reshape_copy(&[2, 3]).unwrap();
    /// let view = a.view();
    /// let argmin_axis_1 = view.argmin_axis(1).unwrap(); // argmin over columns
    /// assert_eq!(argmin_axis_1.to_vec(), vec![0.0, 1.0]); // row0 min at col0, row1 min at col1
    /// ```
    fn argmin_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }
        if self.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let out_shape = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .collect::<Vec<usize>>();
        let out_len = out_shape.iter().product::<usize>();
        let mut out_data: Vec<f64> = vec![0.0; out_len]; // store the index (as f64)
        let mut out_vals: Vec<f64> = vec![f64::INFINITY; out_len];
        let out_strides = compute_strides(&out_shape, self.order());

        for in_flat in 0..self.length() {
            let in_coords = unravel_index(in_flat, &self.shape, self.order)?;
            let out_coords = in_coords
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &c)| c)
                .collect::<Vec<usize>>();
            let out_flat = out_coords
                .iter()
                .enumerate()
                .fold(0, |idx, (i, &c)| idx + c * out_strides[i]);

            let val = self.get_flat(in_flat)?;
            let coord_on_axis = in_coords[axis] as f64;
            if *val < out_vals[out_flat] {
                out_vals[out_flat] = *val;
                out_data[out_flat] = coord_on_axis;
            }
        }

        Ok(Array {
            storage: out_data,
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            order: self.order,
        })
    }

    /// Returns the flat index of the first occurrence of the maximum value.
    ///
    /// The index is returned as a 0‑dimensional array containing a `f64`.
    ///
    /// # Errors
    /// `ArrayError::EmptyArray` if the array is empty.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ReduceOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![3.0, -1.0, 4.0, 4.0]);
    /// let v = a.view();
    /// let idx = v.argmax().unwrap();
    /// assert_eq!(idx.to_scalar().unwrap(), 2.0); // first occurrence at index 2
    /// ```
    fn argmax(&self) -> Result<Self::Output, Self::Error> {
        if self.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let (idx, _) =
            self.iter()
                .enumerate()
                .fold((0, f64::NEG_INFINITY), |(best_idx, best_val), (i, &x)| {
                    if x > best_val {
                        (i, x)
                    } else {
                        (best_idx, best_val)
                    }
                });
        Ok(Array::from_scalar(idx as f64))
    }

    /// Returns the indices (flattened along the reduced axis) of the maximum values along the given axis.
    ///
    /// The output array has the same shape as the input with the specified axis removed. Each entry
    /// contains the linear index (within the reduced axis) of the first occurrence of the maximum
    /// value along that axis.
    ///
    /// # Arguments
    /// * `axis` – The axis along which to find the index of the maximum (0‑based).
    ///
    /// # Errors
    /// * `ArrayError::AxisOutOfBounds` – if `axis >= ndim()`.
    /// * `ArrayError::EmptyArray` – if the input is empty.
    ///
    /// # Returns
    /// An `Array<f64, Vec<f64>>` where each element is the index (as `f64`) of the maximum
    /// along the reduced axis.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ShapeOps, ReduceOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 5.0, 2.0, 3.0, 0.0, 4.0])
    ///     .reshape_copy(&[2, 3]).unwrap();
    /// let view = a.view();
    /// let argmax_axis_1 = view.argmax_axis(1).unwrap(); // argmax over columns
    /// assert_eq!(argmax_axis_1.to_vec(), vec![1.0, 2.0]); // row0 max at col0, row1 max at col1
    /// ```
    fn argmax_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }
        if self.is_empty() {
            return Err(ArrayError::EmptyArray);
        }

        let out_shape = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .collect::<Vec<usize>>();
        let out_len = out_shape.iter().product::<usize>();
        let mut out_data: Vec<f64> = vec![0.0; out_len]; // store the index (as f64)
        let mut out_vals: Vec<f64> = vec![f64::NEG_INFINITY; out_len];
        let out_strides = compute_strides(&out_shape, self.order());

        for in_flat in 0..self.length() {
            let in_coords = unravel_index(in_flat, &self.shape, self.order)?;
            let out_coords = in_coords
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &c)| c)
                .collect::<Vec<usize>>();
            let out_flat = out_coords
                .iter()
                .enumerate()
                .fold(0, |idx, (i, &c)| idx + c * out_strides[i]);

            let val = self.get_flat(in_flat)?;
            let coord_on_axis = in_coords[axis] as f64;
            if *val > out_vals[out_flat] {
                out_vals[out_flat] = *val;
                out_data[out_flat] = coord_on_axis;
            }
        }

        Ok(Array {
            storage: out_data,
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            order: self.order,
        })
    }

    /// Returns `1.0` if any element in the array equals the given value, otherwise `0.0`.
    ///
    /// # Arguments
    /// * `value` – The value to compare against (converted to `f64`).
    ///
    /// # Returns
    /// A 0‑dimensional `Array<f64, Vec<f64>>` containing `1.0` (true) or `0.0` (false).
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ReduceOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let view = a.view();
    /// let has_two = view.any(2.0).unwrap();
    /// assert_eq!(has_two.to_scalar().unwrap(), 1.0);
    /// let has_five = view.any(5.0).unwrap();
    /// assert_eq!(has_five.to_scalar().unwrap(), 0.0);
    /// ```
    fn any(&self, value: f64) -> Result<Self::Output, Self::Error> {
        let result = self.iter().any(|&x| x == value);
        Ok(Array::from_scalar(if result { 1.0 } else { 0.0 }))
    }

    /// Reduces along the specified axis, returning `1.0` for each position where **any**
    /// element along that axis equals the given value, otherwise `0.0`.
    ///
    /// The output has one fewer dimension than the input.
    ///
    /// # Arguments
    /// * `value` – The value to compare against.
    /// * `axis` – The axis along which to reduce (0‑based).
    ///
    /// # Returns
    /// An `Array<f64, Vec<f64>>` of the reduced shape with entries `1.0` (true) or `0.0` (false).
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ReduceOps, ShapeOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0])
    ///     .reshape_copy(&[2, 2]).unwrap();
    /// let any_axis_0 = a.any_axis(2.0, 0).unwrap(); // any in column0? column0: [1.0, 4.0] -> 4.0 != 2.0; column1: [2.0, 3.0] -> 2.0 == 2.0
    /// assert_eq!(any_axis_0.to_vec(), vec![0.0, 1.0]); // shape [2]
    /// ```
    fn any_axis(&self, value: f64, axis: usize) -> Result<Self::Output, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }

        let out_shape = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .collect::<Vec<usize>>();
        let out_len = out_shape.iter().product::<usize>();
        let mut out_data: Vec<f64> = vec![0.0; out_len]; // store the index (as f64)
        let out_strides = compute_strides(&out_shape, self.order());

        for in_flat in 0..self.length() {
            let in_coords = unravel_index(in_flat, &self.shape, self.order)?;
            let out_coords = in_coords
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &c)| c)
                .collect::<Vec<usize>>();
            let out_flat = out_coords
                .iter()
                .enumerate()
                .fold(0, |idx, (i, &c)| idx + c * out_strides[i]);

            let x = self.get_flat(in_flat)?;
            if *x == value {
                out_data[out_flat] = 1.0;
            }
        }

        Ok(Array {
            storage: out_data,
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            order: self.order,
        })
    }

    /// Returns `1.0` if all elements in the array equal the given value, otherwise `0.0`.
    ///
    /// # Arguments
    /// * `value` – The value to compare against (converted to `f64`).
    ///
    /// # Returns
    /// A 0‑dimensional `Array<f64, Vec<f64>>` containing `1.0` (true) or `0.0` (false).
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ReduceOps, ConvertOps};
    ///
    /// let a = Array::ones(&[2, 2]);
    /// let view = a.view();
    /// let all_one = view.all(1.0).unwrap();
    /// assert_eq!(all_one.to_scalar().unwrap(), 1.0);
    /// ```
    fn all(&self, value: f64) -> Result<Self::Output, Self::Error> {
        let result = self.iter().all(|&x| x == value);
        Ok(Array::from_scalar(if result { 1.0 } else { 0.0 }))
    }

    /// Reduces along the specified axis, returning `1.0` for each position where **all**
    /// elements along that axis equal the given value, otherwise `0.0`.
    ///
    /// The output has one fewer dimension than the input.
    ///
    /// # Arguments
    /// * `value` – The value to compare against.
    /// * `axis` – The axis along which to reduce (0‑based).
    ///
    /// # Returns
    /// An `Array<f64, Vec<f64>>` of the reduced shape with entries `1.0` (true) or `0.0` (false).
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ReduceOps, ShapeOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 1.0, 1.0, 2.0])
    ///     .reshape_copy(&[2, 2]).unwrap();
    /// let view = a.view();
    /// let all_axis_1 = view.all_axis(1.0, 1).unwrap(); // row0: both 1 -> true; row1: [1.0, 2.0] both 1.0? no
    /// assert_eq!(all_axis_1.to_vec(), vec![1.0, 0.0]); // shape [2]
    /// ```
    fn all_axis(&self, value: f64, axis: usize) -> Result<Self::Output, Self::Error> {
        if axis >= self.ndim() {
            return Err(ArrayError::AxisOutOfBounds);
        }

        let out_shape = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .collect::<Vec<usize>>();
        let out_len = out_shape.iter().product::<usize>();
        let mut out_data: Vec<f64> = vec![1.0; out_len]; // store the index (as f64)
        let out_strides = compute_strides(&out_shape, self.order());

        for in_flat in 0..self.length() {
            let in_coords = unravel_index(in_flat, &self.shape, self.order)?;
            let out_coords = in_coords
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &c)| c)
                .collect::<Vec<usize>>();
            let out_flat = out_coords
                .iter()
                .enumerate()
                .fold(0, |idx, (i, &c)| idx + c * out_strides[i]);

            let x = self.get_flat(in_flat)?;
            if *x != value {
                out_data[out_flat] = 0.0;
            }
        }

        Ok(Array {
            storage: out_data,
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            order: self.order,
        })
    }
}

impl ReduceOps for Array<f64, Vec<f64>> {
    type Output = Array<f64, Vec<f64>>;

    /// Returns the sum of all elements in the array as a 0‑dimensional array.
    ///
    /// See [`ArrayView::sum`] for details.
    fn sum(&self) -> Result<Self::Output, Self::Error> {
        self.view().sum()
    }

    /// Computes the sum of elements along the specified axis.
    ///
    /// See [`ArrayView::sum_axis`] for details.
    fn sum_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        self.view().sum_axis(axis)
    }

    /// Returns the arithmetic mean of all elements as a 0‑dimensional array.
    ///
    /// See [`ArrayView::mean`] for details.
    fn mean(&self) -> Result<Self::Output, Self::Error> {
        self.view().mean()
    }

    /// Computes the arithmetic mean (average) of elements along the specified axis.
    ///
    /// See [`ArrayView::mean_axis`] for details.
    fn mean_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        self.view().mean_axis(axis)
    }

    /// Returns the sample variance of all elements (unbiased) as a 0‑dimensional array.
    ///
    /// See [`ArrayView::var`] for details.
    fn var(&self) -> Result<Self::Output, Self::Error> {
        self.view().var()
    }

    /// Computes the sample variance of elements along the specified axis.
    ///
    /// See [`ArrayView::var_axis`] for details.
    fn var_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        self.view().var_axis(axis)
    }

    /// Returns the sample standard deviation of all elements as a 0‑dimensional array.
    ///
    /// See [`ArrayView::std`] for details.
    fn std(&self) -> Result<Self::Output, Self::Error> {
        self.view().std()
    }

    /// Computes the sample standard deviation of elements along the specified axis.
    ///
    /// See [`ArrayView::std_axis`] for details.
    fn std_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        self.view().std_axis(axis)
    }

    /// Returns the product of all elements as a 0‑dimensional array.
    ///
    /// See [`ArrayView::prod`] for details.
    fn prod(&self) -> Result<Self::Output, Self::Error> {
        self.view().prod()
    }

    /// Computes the product of elements along the specified axis.
    ///
    /// See [`ArrayView::prod_axis`] for details.
    fn prod_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        self.view().prod_axis(axis)
    }

    /// Returns the minimum value of all elements as a 0‑dimensional array.
    ///
    /// See [`ArrayView::min`] for details.
    fn min(&self) -> Result<Self::Output, Self::Error> {
        self.view().min()
    }

    /// Computes the minimum of elements along the specified axis.
    ///
    /// See [`ArrayView::min_axis`] for details.
    fn min_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        self.view().min_axis(axis)
    }

    /// Returns the maximum value of all elements as a 0‑dimensional array.
    ///
    /// See [`ArrayView::max`] for details.
    fn max(&self) -> Result<Self::Output, Self::Error> {
        self.view().max()
    }

    /// Computes the maximum of elements along the specified axis.
    ///
    /// See [`ArrayView::max_axis`] for details.
    fn max_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        self.view().max_axis(axis)
    }

    /// Returns the flat index of the first occurrence of the minimum value.
    ///
    /// See [`ArrayView::argmin`] for details.
    fn argmin(&self) -> Result<Self::Output, Self::Error> {
        self.view().argmin()
    }

    /// Returns the indices (flattened along the reduced axis) of the minimum values along the given axis.
    ///
    /// See [`ArrayView::argmin_axis`] for details.
    fn argmin_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        self.view().argmin_axis(axis)
    }

    /// Returns the flat index of the first occurrence of the maximum value.
    ///
    /// See [`ArrayView::argmax`] for details.
    fn argmax(&self) -> Result<Self::Output, Self::Error> {
        self.view().argmax()
    }

    /// Returns the indices (flattened along the reduced axis) of the maximum values along the given axis.
    ///
    /// See [`ArrayView::argmax_axis`] for details.
    fn argmax_axis(&self, axis: usize) -> Result<Self::Output, Self::Error> {
        self.view().argmax_axis(axis)
    }

    /// Returns `1.0` if any element in the array equals the given value, otherwise `0.0`.
    ///
    /// See [`ArrayView::any`] for details.
    fn any(&self, value: f64) -> Result<Self::Output, Self::Error> {
        self.view().any(value)
    }

    /// Reduces along the specified axis, returning `1.0` for each position where **any**
    /// element along that axis equals the given value, otherwise `0.0`.
    ///
    /// See [`ArrayView::any_axis`] for details.
    fn any_axis(&self, value: f64, axis: usize) -> Result<Self::Output, Self::Error> {
        self.view().any_axis(value, axis)
    }

    /// Returns `1.0` if all elements in the array equal the given value, otherwise `0.0`.
    ///
    /// See [`ArrayView::all`] for details.
    fn all(&self, value: f64) -> Result<Self::Output, Self::Error> {
        self.view().all(value)
    }

    /// Reduces along the specified axis, returning `1.0` for each position where **all**
    /// elements along that axis equal the given value, otherwise `0.0`.
    ///
    /// See [`ArrayView::all_axis`] for details.
    fn all_axis(&self, value: f64, axis: usize) -> Result<Self::Output, Self::Error> {
        self.view().all_axis(value, axis)
    }
}
