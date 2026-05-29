// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use crate::graph::node::NodeIdLike;

pub trait EdgeLike: PartialEq {
    fn from(&self) -> impl NodeIdLike;
    fn to(&self) -> impl NodeIdLike;
}
