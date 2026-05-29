// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::graph::node::NodeLike;

use crate::graph::error::NodeError;

#[derive(PartialEq)]
pub struct Node {
    pub(crate) id: uuid::Uuid,
}

impl NodeLike for Node {
    type NodeId = uuid::Uuid;
    type Error = NodeError;

    fn id(&self) -> Self::NodeId {
        self.id
    }
}
