// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use crate::graph::core::GraphLike;

pub trait ComputeGraphLike: GraphLike {
    fn evaluate(&self, node_id: &Self::NodeId) -> Result<(), Self::Error>;
    fn evaluate_all(&self) -> Result<(), Self::Error>;
}
