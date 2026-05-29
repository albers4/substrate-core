// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use crate::graph::{
    edge::EdgeLike,
    node::{NodeIdLike, NodeLike},
};

pub trait GraphLike {
    type NodeId: NodeIdLike;
    type Node: NodeLike;
    type Edge: EdgeLike;
    type Error;
}

pub trait GraphViewLike {
    type Owned: GraphLike;

    fn into_owned(self) -> Self::Owned;
}
