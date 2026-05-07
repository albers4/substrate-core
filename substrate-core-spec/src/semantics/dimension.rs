// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::{fmt::Debug, marker::PhantomData};

pub trait Dimension: PartialEq + Eq + Debug {
    const M: i8;
    const KG: i8;
    const S: i8;
    const A: i8;
    const K: i8;
    const MOL: i8;
    const CD: i8;

    fn to_string() -> &'static str {
        let mut parts = Vec::new();

        if Self::M != 0 {
            parts.push(format!("m^{}", Self::M));
        }
        if Self::KG != 0 {
            parts.push(format!("kg^{}", Self::KG));
        }
        if Self::S != 0 {
            parts.push(format!("s^{}", Self::S));
        }
        if Self::A != 0 {
            parts.push(format!("A^{}", Self::A));
        }
        if Self::K != 0 {
            parts.push(format!("K^{}", Self::K));
        }
        if Self::MOL != 0 {
            parts.push(format!("mol^{}", Self::MOL));
        }
        if Self::CD != 0 {
            parts.push(format!("cd^{}", Self::CD));
        }

        if parts.is_empty() {
            "?"
        } else {
            Box::leak(parts.join(" ").into_boxed_str())
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct Dimensionless;
impl Dimension for Dimensionless {
    const M: i8 = 0;
    const KG: i8 = 0;
    const S: i8 = 0;
    const A: i8 = 0;
    const K: i8 = 0;
    const MOL: i8 = 0;
    const CD: i8 = 0;
}

#[derive(PartialEq, Eq, Debug)]
pub struct Length;
impl Dimension for Length {
    const M: i8 = 1;
    const KG: i8 = 0;
    const S: i8 = 0;
    const A: i8 = 0;
    const K: i8 = 0;
    const MOL: i8 = 0;
    const CD: i8 = 0;
}

#[derive(PartialEq, Eq, Debug)]
pub struct Mass;
impl Dimension for Mass {
    const M: i8 = 0;
    const KG: i8 = 1;
    const S: i8 = 0;
    const A: i8 = 0;
    const K: i8 = 0;
    const MOL: i8 = 0;
    const CD: i8 = 0;
}

#[derive(PartialEq, Eq, Debug)]
pub struct Time;
impl Dimension for Time {
    const M: i8 = 0;
    const KG: i8 = 0;
    const S: i8 = 1;
    const A: i8 = 0;
    const K: i8 = 0;
    const MOL: i8 = 0;
    const CD: i8 = 0;
}

#[derive(PartialEq, Eq, Debug)]
pub struct Current;
impl Dimension for Current {
    const M: i8 = 0;
    const KG: i8 = 0;
    const S: i8 = 0;
    const A: i8 = 1;
    const K: i8 = 0;
    const MOL: i8 = 0;
    const CD: i8 = 0;
}

#[derive(PartialEq, Eq, Debug)]
pub struct Temperature;
impl Dimension for Temperature {
    const M: i8 = 0;
    const KG: i8 = 0;
    const S: i8 = 0;
    const A: i8 = 0;
    const K: i8 = 1;
    const MOL: i8 = 0;
    const CD: i8 = 0;
}

#[derive(PartialEq, Eq, Debug)]
pub struct AmountOfSubstance;
impl Dimension for AmountOfSubstance {
    const M: i8 = 0;
    const KG: i8 = 0;
    const S: i8 = 0;
    const A: i8 = 0;
    const K: i8 = 0;
    const MOL: i8 = 1;
    const CD: i8 = 0;
}

#[derive(PartialEq, Eq, Debug)]
pub struct LuminousIntensity;
impl Dimension for LuminousIntensity {
    const M: i8 = 0;
    const KG: i8 = 0;
    const S: i8 = 0;
    const A: i8 = 0;
    const K: i8 = 0;
    const MOL: i8 = 0;
    const CD: i8 = 1;
}

#[derive(PartialEq, Eq, Debug)]
pub struct MulDim<A: Dimension, B: Dimension>(PhantomData<(A, B)>);
impl<A: Dimension, B: Dimension> Dimension for MulDim<A, B> {
    const M: i8 = A::M + B::M;
    const KG: i8 = A::KG + B::KG;
    const S: i8 = A::S + B::S;
    const A: i8 = A::A + B::A;
    const K: i8 = A::K + B::K;
    const MOL: i8 = A::MOL + B::MOL;
    const CD: i8 = A::CD + B::CD;
}

#[derive(PartialEq, Eq, Debug)]
pub struct DivDim<A: Dimension, B: Dimension>(PhantomData<(A, B)>);
impl<A: Dimension, B: Dimension> Dimension for DivDim<A, B> {
    const M: i8 = A::M - B::M;
    const KG: i8 = A::KG - B::KG;
    const S: i8 = A::S - B::S;
    const A: i8 = A::A - B::A;
    const K: i8 = A::K - B::K;
    const MOL: i8 = A::MOL - B::MOL;
    const CD: i8 = A::CD - B::CD;
}
