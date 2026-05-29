// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::graph::{edge::EdgeLike, node::NodeIdLike};

#[derive(PartialEq)]
pub struct Edge(uuid::Uuid, uuid::Uuid);

impl Edge {
    pub fn new(from: uuid::Uuid, to: uuid::Uuid) -> Self {
        Self(from, to)
    }
}

impl EdgeLike for Edge {
    fn from(&self) -> impl NodeIdLike {
        self.0
    }

    fn to(&self) -> impl NodeIdLike {
        self.1
    }
}
