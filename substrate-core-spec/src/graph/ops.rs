// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::collections::{HashMap, HashSet};

use crate::graph::core::GraphLike;

pub trait AccessOps: GraphLike {
    fn get_node(&self, id: &Self::NodeId) -> Option<&Self::Node>;
    fn get_edge(&self, from: &Self::NodeId, to: &Self::NodeId) -> Option<&Self::Edge>;

    fn nodes(&self) -> impl Iterator<Item = (&Self::NodeId, &Self::Node)>;
    fn edges(&self) -> impl Iterator<Item = (&Self::NodeId, &Self::NodeId, &Self::Edge)>;

    fn contains_node(&self, id: &Self::NodeId) -> bool;
    fn contains_edge(&self, from: &Self::NodeId, to: &Self::NodeId) -> bool;
}

pub trait AccessOpsMut: AccessOps {
    fn add_node(&mut self, data: Self::Node) -> Self::NodeId;
    fn add_edge(
        &mut self,
        from: &Self::NodeId,
        to: &Self::NodeId,
        data: Self::Edge,
    ) -> Result<(), Self::Error>;

    fn remove_node(&mut self, id: &Self::NodeId) -> Option<Self::Node>;
    fn remove_edge(&mut self, from: &Self::NodeId, to: &Self::NodeId) -> Option<Self::Edge>;

    fn node_mut(&mut self, id: &Self::NodeId) -> Option<&mut Self::Node>;
    fn edge_mut(&mut self, from: &Self::NodeId, to: &Self::NodeId) -> Option<&mut Self::Edge>;
}

pub trait AdvancedAccess: AccessOps {
    fn neighbors(&self, id: &Self::NodeId) -> impl Iterator<Item = &Self::NodeId>;
    fn out_neighbors(&self, id: &Self::NodeId) -> impl Iterator<Item = &Self::NodeId>;
    fn in_neighbors(&self, id: &Self::NodeId) -> impl Iterator<Item = &Self::NodeId>;
    fn all_node_ids(&self) -> Vec<Self::NodeId>;
}

pub trait GraphAlgorithms: AccessOps + AdvancedAccess {
    fn topological_sort(&self) -> Result<Vec<&Self::NodeId>, Self::Error>;
    fn has_self_loops(&self) -> bool;
    fn is_simple(&self) -> bool;
    fn has_cycle(&self) -> bool;
    fn is_dag(&self) -> bool;
    fn is_connected(&self) -> bool;
    fn is_bipartite(&self) -> bool;
    fn is_strongly_connect(&self) -> bool;
    fn reachable_from(&self, start: &Self::NodeId) -> HashSet<Self::NodeId>;
    fn reverse_reachable_from(&self, start: &Self::NodeId) -> HashSet<Self::NodeId>;
    fn strongly_connected_components(&self) -> Vec<Vec<Self::NodeId>>;
    fn shorted_path_unweighted(
        &self,
        from: &Self::NodeId,
        to: &Self::NodeId,
    ) -> Option<Vec<Self::NodeId>>;
    fn degree_distribution(&self) -> HashMap<usize, usize>;
    fn transitive_closure(&self) -> HashMap<Self::NodeId, Vec<Self::NodeId>>;
}
