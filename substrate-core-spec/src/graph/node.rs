// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::hash::Hash;
use uuid::Uuid;

pub trait NodeIdLike: PartialEq + Eq + Hash + Copy {}

impl NodeIdLike for Uuid {}

pub trait NodeValueLike: PartialEq + std::fmt::Debug {}

pub trait NodeLike: PartialEq {
    type NodeId: NodeIdLike;
    type Error;

    fn id(&self) -> Self::NodeId;
}

pub trait ComputeNodeLike: NodeLike {
    type Value: NodeValueLike;

    fn in_shape(&self) -> Vec<usize>;
    fn out_shape(&self) -> Vec<usize>;

    fn in_count(&self) -> usize;

    fn evaluate(&self) -> Result<Self::Value, Self::Error>;
}
