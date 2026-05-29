// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use rand::RngExt;
use rand_distr::StandardNormal;
use substrate_core_spec::array::ops::{AccessOps, InitOps};

use crate::{Array, array::error::ArrayError};

impl InitOps for Array<f64, Vec<f64>> {
    type Output = Self;

    /// Creates a new array filled with uniformly distributed random numbers in the range `[0, 1)`.
    ///
    /// The function uses the `rand` crate’s default random number generator.
    ///
    /// # Arguments
    /// * `shape` – The shape of the resulting array.
    ///
    /// # Returns
    /// `Ok(Array<f64, Vec<f64>>)` with the given shape, filled with random values.
    ///
    /// # Errors
    /// * `ArrayError::InvalidShape` – if the shape is invalid (e.g., contains zero dimensions,
    ///   or the product of dimensions overflows `usize`). (Propagated from `from_vec_with_shape`.)
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::InitOps;
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::rand(&[2, 3]).unwrap();
    /// assert_eq!(a.shape(), &[2, 3]);
    /// ```
    fn rand(shape: &[usize]) -> Result<Self::Output, Self::Error> {
        let size = shape.iter().product::<usize>();
        let mut rng = rand::rng();
        let data = (0..size).map(|_| rng.random()).collect();
        Array::from_vec_with_shape(data, shape)
    }

    /// Creates a new array filled with random numbers from the standard normal distribution
    /// (mean = 0, variance = 1).
    ///
    /// The function uses the `rand` and `rand_distr` crates (`StandardNormal` distribution).
    ///
    /// # Arguments
    /// * `shape` – The shape of the resulting array.
    ///
    /// # Returns
    /// `Ok(Array<f64, Vec<f64>>)` with the given shape, filled with standard normal deviates.
    ///
    /// # Errors
    /// * `ArrayError::InvalidShape` – if the shape is invalid (e.g., contains zero dimensions,
    ///   or the product of dimensions overflows `usize`). (Propagated from `from_vec_with_shape`.)
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::InitOps;
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::randn(&[2, 3]).unwrap();
    /// assert_eq!(a.shape(), &[2, 3]);
    /// ```
    fn randn(shape: &[usize]) -> Result<Self::Output, Self::Error> {
        let size = shape.iter().product::<usize>();
        let mut rng = rand::rng();
        let data = (0..size).map(|_| rng.sample(StandardNormal)).collect();
        Array::from_vec_with_shape(data, shape)
    }

    /// Creates an `n x n` identity matrix with ones on the diagonal and zeros elsewhere.
    ///
    /// # Arguments
    /// * `n` – Size of the square matrix.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` of shape `[n, n]`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let i = Array::eye(3).unwrap();
    /// assert_eq!(i.shape(), &[3, 3]);
    /// assert_eq!(i.to_vec(), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    /// ```
    fn eye(n: usize) -> Result<Self::Output, Self::Error> {
        let shape = vec![n, n];
        let size = n * n;
        let mut data = vec![0.0; size];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Array::from_vec_with_shape(data, &shape)
    }

    /// Creates a diagonal matrix from a 1‑D array.
    ///
    /// # Arguments
    /// * `diag` – A 1‑D array containing the diagonal elements.
    ///
    /// # Returns
    /// A square matrix of shape `[len(diag), len(diag)]` with `diag` on the main diagonal.
    ///
    /// # Errors
    /// * `ArrayError::DimensionMismatch` – if `diag.ndim() != 1`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let d = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// let m = Array::diag(&d).unwrap();
    /// assert_eq!(m.shape(), &[3, 3]);
    /// assert_eq!(m.to_vec(), vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    /// ```
    fn diag(
        diag: &impl AccessOps<Item = Self::Item, Error = Self::Error>,
    ) -> Result<Self::Output, Self::Error> {
        if diag.ndim() != 1 {
            return Err(ArrayError::DimensionMismatch);
        }

        let n = diag.length();
        let shape = vec![n, n];
        let size = n * n;
        let mut data = vec![0.0; size];
        for i in 0..n {
            let val = *diag.get_flat(i)?;
            data[i * n + i] = val;
        }
        Array::from_vec_with_shape(data, &shape)
    }

    /// Creates an array of given shape filled with a constant value.
    ///
    /// # Arguments
    /// * `shape` – Desired shape.
    /// * `value` – Constant value to fill.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` where every element equals `value`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::full(&[2, 3], 5.0).unwrap();
    /// assert_eq!(a.shape(), &[2, 3]);
    /// assert_eq!(a.to_vec(), vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
    /// ```
    fn full(shape: &[usize], value: Self::Item) -> Result<Self::Output, Self::Error> {
        let size = shape.iter().product::<usize>();
        let data = vec![value; size];
        Array::from_vec_with_shape(data, shape)
    }

    /// Returns a 1‑dimensional array with evenly spaced values within a given interval.
    ///
    /// The values are generated in the half‑open interval `[start, end)` with step `step`.
    ///
    /// # Arguments
    /// * `start` – Start value (inclusive).
    /// * `end` – End value (exclusive).
    /// * `step` – Step size (must be positive).
    ///
    /// # Returns
    /// `Result<Self::Output, Self::Error>` containing a 1‑D array.
    ///
    /// # Errors
    /// * `ArrayError::InvalidStep` – if `step <= 0.0`.
    /// * `ArrayError::EmptyArray` – if no elements would be generated (e.g., `start >= end` and `step > 0`).
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps};
    ///
    /// let a = Array::arange(0.0, 5.0, 1.0).unwrap();
    /// assert_eq!(a.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    /// ```
    fn arange(
        start: Self::Item,
        end: Self::Item,
        step: Self::Item,
    ) -> Result<Self::Output, Self::Error> {
        if step <= 0.0 {
            return Err(ArrayError::InvalidStep);
        }

        let n = ((end - start) / step).ceil() as usize;
        let mut data = Vec::with_capacity(n);
        let mut val = start;
        for _ in 0..n {
            data.push(val);
            val += step;
            if val >= end {
                break;
            }
        }
        let len = data.len();
        Array::from_vec_with_shape(data, &[len])
    }

    /// Returns a 1‑dimensional array with values spaced logarithmically between `10^a` and `10^b`.
    ///
    /// The number of points `n` must be at least 2. The values are computed as `10^(a + i * step)`,
    /// where `step = (b - a) / (n - 1)`. If a base different from 10 is desired, use a separate method.
    ///
    /// # Arguments
    /// * `a` – The exponent for the first value (`10^a`).
    /// * `b` – The exponent for the last value (`10^b`).
    /// * `n` – Number of points (must be >= 2).
    ///
    /// # Returns
    /// `Result<Self::Output, Self::Error>` – a 1‑D array of length `n`.
    ///
    /// # Errors
    /// * `ArrayError::InvalidParameter` – if `n < 2`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps};
    ///
    /// let a = Array::logspace(0.0, 2.0, 3).unwrap();
    /// assert_eq!(a.to_vec(), vec![1.0, 10.0, 100.0]);
    /// ```
    fn logspace(a: Self::Item, b: Self::Item, n: usize) -> Result<Self::Output, Self::Error> {
        if n < 2 {
            return Err(ArrayError::InvalidParameter);
        }

        let step = (b - a) / (n - 1) as f64;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let exponent = a + i as f64 * step;
            data.push(10.0_f64.powf(exponent));
        }
        Array::from_vec_with_shape(data, &[n])
    }

    /// Creates a new array filled with zeros of the given shape.
    ///
    /// # Arguments
    /// * `shape` - A slice specifying the dimensions of the array.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` where every element is `0.0`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ArrayLike;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps};
    ///
    /// let a = Array::zeros(&[2, 3]);
    /// assert_eq!(a.shape(), &[2, 3]);
    /// assert_eq!(a.to_vec(), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    /// ```
    fn zeros(shape: &[usize]) -> Self::Output {
        Self::from_scalar_with_shape(0.0f64, shape)
    }

    /// Creates a new array filled with ones of the given shape.
    ///
    /// # Arguments
    /// * `shape` - A slice specifying the dimensions of the array.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` where every element is `1.0`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps};
    ///
    /// let a = Array::ones(&[2, 2]);
    /// assert_eq!(a.to_vec(), vec![1.0, 1.0, 1.0, 1.0]);
    /// ```
    fn ones(shape: &[usize]) -> Self::Output {
        Self::from_scalar_with_shape(1.0f64, shape)
    }

    /// Creates a 1‑dimensional array from a vector of elements.
    ///
    /// # Arguments
    /// * `vec` - The vector containing the element values.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with shape `[vec.len()]`, row‑major strides,
    /// and default memory order.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps};
    /// use substrate_core_spec::array::ArrayLike;
    ///
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(a.shape(), &[3]);
    /// assert_eq!(a.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    fn from_vec(vec: Vec<Self::Item>) -> Self::Output {
        Self::from_vec(vec)
    }

    /// Creates a 1‑dimensional array with `n` linearly spaced elements between `a` and `b`.
    ///
    /// # Arguments
    /// * `a` - Start value (inclusive).
    /// * `b` - End value (inclusive).
    /// * `n` - Number of elements.
    ///
    /// # Returns
    /// `Ok(Array<f64, Vec<f64>>)` if `n > 0`, otherwise `Err(ArrayError::ArrayFromLinspaceError)`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps};
    ///
    /// let a = Array::linspace(0.0, 1.0, 5).unwrap();
    /// assert_eq!(a.to_vec(), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    ///
    /// assert!(Array::linspace(0.0, 1.0, 0).is_err());
    /// ```
    fn linspace(a: Self::Item, b: Self::Item, n: usize) -> Result<Self::Output, Self::Error> {
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

    /// Creates a new array by applying a function to each logical index.
    ///
    /// # Arguments
    /// * `shape` - The shape of the resulting array.
    /// * `f` - A closure that receives a slice of indices (one per axis) and returns the element value.
    ///
    /// # Returns
    /// `Ok(Array<f64, Vec<f64>>)` on success, or an error if the shape is invalid (e.g., zero in any dimension).
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, ConvertOps};
    ///
    /// // Create a 2x2 array where element (i,j) = i + j
    /// let a = Array::from_fn(&[2, 2], |idx| (idx[0] + idx[1]) as f64).unwrap();
    /// assert_eq!(a.to_vec(), vec![0.0, 1.0, 1.0, 2.0]);
    ///
    /// // Scalar (0‑dimensional) array
    /// let b = Array::from_fn(&[], |_| 42.0).unwrap();
    /// assert_eq!(b.to_scalar().unwrap(), 42.0);
    /// ```
    fn from_fn<F>(shape: &[usize], mut f: F) -> Result<Self::Output, Self::Error>
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
}
