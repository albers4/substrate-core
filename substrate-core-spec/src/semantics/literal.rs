// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use crate::semantics::{
    arithmetic::Arithmetic,
    dimension::{
        AmountOfSubstance, Current, Dimensionless, Length, LuminousIntensity, Mass, MulDim,
        Temperature, Time,
    },
    quantity::Quantity,
};

pub trait LiteralUnit<T: Arithmetic> {
    fn dimensionless(self) -> Quantity<T, Dimensionless>;

    fn m(self) -> Quantity<T, Length>;
    fn m2(self) -> Quantity<T, MulDim<Length, Length>>;
    fn cm(self) -> Quantity<T, Length>;

    fn kg(self) -> Quantity<T, Mass>;

    fn s(self) -> Quantity<T, Time>;

    fn ampere(self) -> Quantity<T, Current>;

    fn kelvin(self) -> Quantity<T, Temperature>;
    fn deg_c(self) -> Quantity<T, Temperature>;

    fn mole(self) -> Quantity<T, AmountOfSubstance>;

    fn candela(self) -> Quantity<T, LuminousIntensity>;
}

impl LiteralUnit<f64> for f64 {
    fn dimensionless(self) -> Quantity<f64, Dimensionless> {
        Quantity::<f64, Dimensionless>::new(self)
    }

    fn m(self) -> Quantity<f64, Length> {
        Quantity::<f64, Length>::new(self)
    }

    fn m2(self) -> Quantity<f64, MulDim<Length, Length>> {
        Quantity::<f64, MulDim<Length, Length>>::new(self)
    }

    fn cm(self) -> Quantity<f64, Length> {
        Quantity::<f64, Length>::new(self / 100.0)
    }

    fn kg(self) -> Quantity<f64, Mass> {
        Quantity::<f64, Mass>::new(self)
    }

    fn s(self) -> Quantity<f64, Time> {
        Quantity::<f64, Time>::new(self)
    }

    fn ampere(self) -> Quantity<f64, Current> {
        Quantity::<f64, Current>::new(self)
    }

    fn kelvin(self) -> Quantity<f64, Temperature> {
        Quantity::<f64, Temperature>::new(self)
    }

    fn deg_c(self) -> Quantity<f64, Temperature> {
        Quantity::<f64, Temperature>::new(self + 273.15)
    }

    fn mole(self) -> Quantity<f64, AmountOfSubstance> {
        Quantity::<f64, AmountOfSubstance>::new(self)
    }

    fn candela(self) -> Quantity<f64, LuminousIntensity> {
        Quantity::<f64, LuminousIntensity>::new(self)
    }
}
