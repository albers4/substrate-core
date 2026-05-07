// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::semantics::{
    dimension::{DivDim, Length, Mass, MulDim, Temperature, Time},
    quantity::Quantity,
};

use crate::semantics::derived_dimensions::Velocity;

pub struct PhysicalConstants;

type GravitationalDimension =
    DivDim<MulDim<MulDim<Length, Length>, Length>, MulDim<Mass, MulDim<Time, Time>>>;
type PlanckDimension = DivDim<MulDim<MulDim<Length, Length>, Mass>, Time>;
type BoltzmannDimension =
    DivDim<DivDim<MulDim<MulDim<Length, Length>, Mass>, MulDim<Time, Time>>, Temperature>;
type StefanBoltzmannDimension = DivDim<
    DivDim<Mass, MulDim<MulDim<Time, Time>, Time>>,
    MulDim<Temperature, MulDim<Temperature, MulDim<Temperature, Temperature>>>,
>;

impl PhysicalConstants {
    /// Speed of light in a vacuum (m1^ s^-1)
    pub fn speed_of_light() -> Quantity<f64, Velocity> {
        Quantity::new(299792458.0)
    }

    /// Gravitational constant (m^3 kg^-1 s^-2)
    pub fn gravitational_constant() -> Quantity<f64, GravitationalDimension> {
        Quantity::new(6.67430e-11)
    }

    /// Planck constant (m^2 kg^1 s^-1)
    pub fn planck_constant() -> Quantity<f64, PlanckDimension> {
        Quantity::new(6.62607015e-34)
    }

    /// Boltzmann constant (m^2 kg^1 s^-2 K^-1)
    pub fn boltzmann_constant() -> Quantity<f64, BoltzmannDimension> {
        Quantity::new(1.380649e-23)
    }

    /// Stefan-Boltzmann constant (kg^1 s^-3 K^-4)
    pub fn stefan_boltzmann_constant() -> Quantity<f64, StefanBoltzmannDimension> {
        Quantity::new(5.670374419e-8)
    }
}
