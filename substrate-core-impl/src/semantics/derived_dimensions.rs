// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::semantics::dimension::{DivDim, Length, Mass, MulDim, Time};

pub type Area = MulDim<Length, Length>;
pub type Velocity = DivDim<Length, Time>;
pub type Acceleration = DivDim<Velocity, Time>;
pub type Force = MulDim<Mass, Acceleration>;
pub type Pressure = DivDim<Force, Area>;
pub type Energy = MulDim<Force, Length>;
pub type Power = DivDim<Energy, Time>;
