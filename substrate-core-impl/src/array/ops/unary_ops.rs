// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use core::ops::Neg;

use substrate_core_spec::array::{
    ArrayLike,
    ops::{ConvertOps, UnaryOps},
};

use crate::{Array, array::ArrayView};

impl<'a> UnaryOps for ArrayView<'a, f64> {
    type Output = Array<f64, Vec<f64>>;

    /// Returns a new array with the absolute value of each element.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `|x|` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![-1.0, -2.0, 3.0]);
    /// let abs_a = a.view().abs().unwrap();
    /// assert_eq!(abs_a.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    fn abs(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.abs()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the negated value of each element.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `-x` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, -2.0, 3.0]);
    /// let neg_a = a.view().neg().unwrap();
    /// assert_eq!(neg_a.to_vec(), vec![-1.0, 2.0, -3.0]);
    /// ```
    fn neg(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.neg()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the square root of each element.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `sqrt(x)` for each input element.
    /// Undefined behaviour for negative inputs (produces NaN).
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![4.0, 9.0, 16.0]);
    /// let sqrt_a = a.view().sqrt().unwrap();
    /// assert_eq!(sqrt_a.to_vec(), vec![2.0, 3.0, 4.0]);
    /// ```
    fn sqrt(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.sqrt()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the exponential (e^x) of each element.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `exp(x)` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![0.0, 1.0, 2.0]);
    /// let result = a.view().exp().unwrap().to_vec();
    /// let expected = vec![1.0, std::f64::consts::E, std::f64::consts::E.powi(2)];
    /// for (r, e) in result.iter().zip(expected.iter()) {
    ///     assert!((r - e).abs() < 1e-12);
    /// }
    /// ```
    fn exp(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.exp()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the natural logarithm of each element.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `ln(x)` for each input element.
    /// Input must be positive; negative inputs produce NaN.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, std::f64::consts::E]);
    /// let ln_a = a.view().ln().unwrap();
    /// assert_eq!(ln_a.to_vec(), vec![0.0, 1.0]);
    /// ```
    fn ln(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.ln()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the logarithm of each element to the given base.
    ///
    /// # Arguments
    /// * `base` - The base of the logarithm.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `log_base(x)` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.0, 10.0, 100.0]);
    /// let log10_a = a.view().log(10.0).unwrap();
    /// assert_eq!(log10_a.to_vec(), vec![0.0, 1.0, 2.0]);
    /// ```
    fn log(&self, base: f64) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.log(base)).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the sine of each element (in radians).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `sin(x)` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI]);
    /// let result = a.view().sin().unwrap().to_vec();
    /// let expected = vec![0.0, 1.0, 0.0];
    /// for (r, e) in result.iter().zip(expected.iter()) {
    ///     assert!((r - e).abs() < 1e-12);
    /// }
    /// ```
    fn sin(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.sin()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the cosine of each element (in radians).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `cos(x)` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI]);
    /// let result = a.view().cos().unwrap().to_vec();
    /// let expected = vec![1.0, 0.0, -1.0];
    /// for (r, e) in result.iter().zip(expected.iter()) {
    ///     assert!((r - e).abs() < 1e-12);
    /// }
    /// ```
    fn cos(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.cos()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the tangent of each element (in radians).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `tan(x)` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![0.0, std::f64::consts::PI / 4.0]);
    /// let result = a.view().tan().unwrap().to_vec();
    /// let expected = vec![0.0, 1.0];
    /// for (r, e) in result.iter().zip(expected.iter()) {
    ///     assert!((r - e).abs() < 1e-12);
    /// }
    /// ```
    fn tan(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.tan()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the arcsine of each element (in radians).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `asin(x)` for each input element.
    /// Input must be in `[-1, 1]`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![0.0, 1.0, -1.0]);
    /// let result = a.view().asin().unwrap().to_vec();
    /// let expected  = vec![0.0, std::f64::consts::PI / 2.0, -std::f64::consts::PI / 2.0];
    /// for (r, e) in result.iter().zip(expected.iter()) {
    ///     assert!((r - e).abs() < 1e-12);
    /// }
    /// ```
    fn asin(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.asin()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the arccosine of each element (in radians).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `acos(x)` for each input element.
    /// Input must be in `[-1, 1]`.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![0.0, 1.0, -1.0]);
    /// let result = a.view().acos().unwrap().to_vec();
    /// let expected = vec![std::f64::consts::PI / 2.0, 0.0, std::f64::consts::PI];
    /// for (r, e) in result.iter().zip(expected.iter()) {
    ///     assert!((r - e).abs() < 1e-12);
    /// }
    /// ```
    fn acos(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.acos()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the arctangent of each element (in radians).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `atan(x)` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![0.0, 1.0, -1.0]);
    /// let result = a.view().atan().unwrap().to_vec();
    /// let expected = vec![0.0, std::f64::consts::PI / 4.0, -std::f64::consts::PI / 4.0];
    /// for (r, e) in result.iter().zip(expected.iter()) {
    ///     assert!((r - e).abs() < 1e-12);
    /// }
    /// ```
    fn atan(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.atan()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the hyperbolic sine of each element.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `sinh(x)` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![0.0, 1.0]);
    /// let result = a.view().sinh().unwrap().to_vec();
    /// let expected = vec![0.0, (std::f64::consts::E - 1.0/std::f64::consts::E) / 2.0];
    /// for (r, e) in result.iter().zip(expected.iter()) {
    ///     assert!((r - e).abs() < 1e-12);
    /// }
    /// ```
    fn sinh(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.sinh()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the hyperbolic cosine of each element.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `cosh(x)` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![0.0, 1.0]);
    /// let result = a.view().cosh().unwrap().to_vec();
    /// let expected = vec![1.0, (std::f64::consts::E + 1.0/std::f64::consts::E) / 2.0];
    /// for (r, e) in result.iter().zip(expected.iter()) {
    ///     assert!((r - e).abs() < 1e-12);
    /// }
    /// ```
    fn cosh(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.cosh()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the hyperbolic tangent of each element.
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `tanh(x)` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![0.0, 1.0]);
    /// let result = a.view().tanh().unwrap().to_vec();
    /// let expected = vec![0.0, (std::f64::consts::E - 1.0/std::f64::consts::E) / (std::f64::consts::E + 1.0/std::f64::consts::E)];
    /// for (r, e) in result.iter().zip(expected.iter()) {
    ///     assert!((r - e).abs() < 1e-12);
    /// }
    /// ```
    fn tanh(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.tanh()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the smallest integer greater than or equal to each element (ceiling).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `ceil(x)` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.2, -1.2, 2.0]);
    /// let ceil_a = a.view().ceil().unwrap();
    /// assert_eq!(ceil_a.to_vec(), vec![2.0, -1.0, 2.0]);
    /// ```
    fn ceil(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.ceil()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the largest integer less than or equal to each element (floor).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `floor(x)` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.2, -1.2, 2.0]);
    /// let floor_a = a.view().floor().unwrap();
    /// assert_eq!(floor_a.to_vec(), vec![1.0, -2.0, 2.0]);
    /// ```
    fn floor(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.floor()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the nearest integer to each element (round half away from zero).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing `round(x)` for each input element.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![1.5, -1.5, 1.2]);
    /// let round_a = a.view().round().unwrap();
    /// assert_eq!(round_a.to_vec(), vec![2.0, -2.0, 1.0]);
    /// ```
    fn round(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.round()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }

    /// Returns a new array with the sign of each element (-1, 0, 1).
    ///
    /// # Returns
    /// A new `Array<f64, Vec<f64>>` with the same shape, containing:
    /// - `-1.0` for negative values, -0.0 or NEG_INFINITY,
    /// - `1.0` for positive values, +0.0 or INFINITY.
    ///
    /// # Examples
    /// ```
    /// use substrate_core_impl::Array;
    /// use substrate_core_spec::array::ops::{InitOps, UnaryOps, ConvertOps};
    ///
    /// let a = Array::from_vec(vec![-2.0, 0.0, 5.0]);
    /// let sign_a = a.view().signum().unwrap();
    /// assert_eq!(sign_a.to_vec(), vec![-1.0, 1.0, 1.0]);
    /// ```
    fn signum(&self) -> Result<Self::Output, Self::Error> {
        let data = self.data.iter().map(|x| x.signum()).collect();
        Array::from_vec_with_shape(data, self.shape())
    }
}

impl UnaryOps for Array<f64, Vec<f64>> {
    type Output = Array<f64, Vec<f64>>;

    /// Returns a new array with the absolute value of each element.
    ///
    /// See [`ArrayView::abs`] for details.
    fn abs(&self) -> Result<Self::Output, Self::Error> {
        self.view().abs()
    }

    /// Returns a new array with the negated value of each element.
    ///
    /// See [`ArrayView::neg`] for details.
    fn neg(&self) -> Result<Self::Output, Self::Error> {
        self.view().neg()
    }

    /// Returns a new array with the square root of each element.
    ///
    /// See [`ArrayView::sqrt`] for details.
    fn sqrt(&self) -> Result<Self::Output, Self::Error> {
        self.view().sqrt()
    }

    /// Returns a new array with the exponential (e^x) of each element.
    ///
    /// See [`ArrayView::exp`] for details.
    fn exp(&self) -> Result<Self::Output, Self::Error> {
        self.view().exp()
    }

    /// Returns a new array with the natural logarithm of each element.
    ///
    /// See [`ArrayView::ln`] for details.
    fn ln(&self) -> Result<Self::Output, Self::Error> {
        self.view().ln()
    }

    /// Returns a new array with the logarithm of each element to the given base.
    ///
    /// See [`ArrayView::log`] for details.
    fn log(&self, base: f64) -> Result<Self::Output, Self::Error> {
        self.view().log(base)
    }

    /// Returns a new array with the sine of each element (in radians).
    ///
    /// See [`ArrayView::sin`] for details.
    fn sin(&self) -> Result<Self::Output, Self::Error> {
        self.view().sin()
    }

    /// Returns a new array with the cosine of each element (in radians).
    ///
    /// See [`ArrayView::cos`] for details.
    fn cos(&self) -> Result<Self::Output, Self::Error> {
        self.view().cos()
    }

    /// Returns a new array with the tangent of each element (in radians).
    ///
    /// See [`ArrayView::tan`] for details.
    fn tan(&self) -> Result<Self::Output, Self::Error> {
        self.view().tan()
    }

    /// Returns a new array with the arcsine of each element (in radians).
    ///
    /// See [`ArrayView::asin`] for details.
    fn asin(&self) -> Result<Self::Output, Self::Error> {
        self.view().asin()
    }

    /// Returns a new array with the arccosine of each element (in radians).
    ///
    /// See [`ArrayView::acos`] for details.
    fn acos(&self) -> Result<Self::Output, Self::Error> {
        self.view().acos()
    }

    /// Returns a new array with the arctangent of each element (in radians).
    ///
    /// See [`ArrayView::atan`] for details.
    fn atan(&self) -> Result<Self::Output, Self::Error> {
        self.view().atan()
    }

    /// Returns a new array with the hyperbolic sine of each element.
    ///
    /// See [`ArrayView::asinh`] for details.
    fn sinh(&self) -> Result<Self::Output, Self::Error> {
        self.view().sinh()
    }

    /// Returns a new array with the hyperbolic cosine of each element.
    ///
    /// See [`ArrayView::cosh`] for details.
    fn cosh(&self) -> Result<Self::Output, Self::Error> {
        self.view().cosh()
    }

    /// Returns a new array with the hyperbolic tangent of each element.
    ///
    /// See [`ArrayView::tanh`] for details.
    fn tanh(&self) -> Result<Self::Output, Self::Error> {
        self.view().tanh()
    }

    /// Returns a new array with the smallest integer greater than or equal to each element (ceiling).
    ///
    /// See [`ArrayView::ceil`] for details.
    fn ceil(&self) -> Result<Self::Output, Self::Error> {
        self.view().ceil()
    }

    /// Returns a new array with the largest integer less than or equal to each element (floor).
    ///
    /// See [`ArrayView::floor`] for details.
    fn floor(&self) -> Result<Self::Output, Self::Error> {
        self.view().floor()
    }

    /// Returns a new array with the nearest integer to each element (round half away from zero).
    ///
    /// See [`ArrayView::round`] for details.
    fn round(&self) -> Result<Self::Output, Self::Error> {
        self.view().round()
    }

    /// Returns a new array with the sign of each element (-1, 0, 1).
    ///
    /// See [`ArrayView::signum`] for details.
    fn signum(&self) -> Result<Self::Output, Self::Error> {
        self.view().signum()
    }
}
