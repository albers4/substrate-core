// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::visit;
use substrate_core_spec::graph::{
    core::GraphLike,
    ops::{AccessOps, AccessOpsMut, AdvancedAccess, GraphAlgorithms},
};

use crate::graph::{edge::Edge, error::GraphError, node::Node};

pub struct Graph {
    pub(crate) nodes: HashMap<<Graph as GraphLike>::NodeId, <Graph as GraphLike>::Node>,
    pub(crate) edges: HashMap<
        (<Graph as GraphLike>::NodeId, <Graph as GraphLike>::NodeId),
        <Graph as GraphLike>::Edge,
    >,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }
}

impl GraphLike for Graph {
    type NodeId = uuid::Uuid;
    type Node = Node;
    type Edge = Edge;
    type Error = GraphError;
}

impl AccessOps for Graph {
    fn get_node(&self, id: &Self::NodeId) -> Option<&Self::Node> {
        self.nodes.get(&id)
    }

    fn get_edge(&self, from: &Self::NodeId, to: &Self::NodeId) -> Option<&Self::Edge> {
        self.edges.get(&(*from, *to))
    }

    fn nodes(&self) -> impl Iterator<Item = (&Self::NodeId, &Self::Node)> {
        self.nodes
            .iter()
            .map(|(node_id, node_data)| (node_id, node_data))
    }

    fn edges(&self) -> impl Iterator<Item = (&Self::NodeId, &Self::NodeId, &Self::Edge)> {
        self.edges
            .iter()
            .map(|(node_ids, edge_data)| (&node_ids.0, &node_ids.1, edge_data))
    }

    fn contains_node(&self, id: &Self::NodeId) -> bool {
        if let Some(_) = self.nodes.get(&id) {
            true
        } else {
            false
        }
    }

    fn contains_edge(&self, from: &Self::NodeId, to: &Self::NodeId) -> bool {
        if let Some(_) = self.edges.get(&(*from, *to)) {
            true
        } else {
            false
        }
    }
}

impl AccessOpsMut for Graph {
    fn add_node(&mut self, data: Self::Node) -> Self::NodeId {
        let node_id = uuid::Uuid::new_v4();
        self.nodes.insert(node_id, data);
        node_id
    }

    fn add_edge(
        &mut self,
        from: &Self::NodeId,
        to: &Self::NodeId,
        data: Self::Edge,
    ) -> Result<(), Self::Error> {
        self.edges.insert((*from, *to), data);
        Ok(())
    }

    fn remove_node(&mut self, id: &Self::NodeId) -> Option<Self::Node> {
        self.nodes.remove(&id)
    }

    fn remove_edge(&mut self, from: &Self::NodeId, to: &Self::NodeId) -> Option<Self::Edge> {
        self.edges.remove(&(*from, *to))
    }

    fn node_mut(&mut self, id: &Self::NodeId) -> Option<&mut Self::Node> {
        self.nodes.get_mut(&id)
    }

    fn edge_mut(&mut self, from: &Self::NodeId, to: &Self::NodeId) -> Option<&mut Self::Edge> {
        self.edges.get_mut(&(*from, *to))
    }
}

impl AdvancedAccess for Graph {
    fn neighbors(&self, id: &Self::NodeId) -> impl Iterator<Item = &Self::NodeId> {
        self.out_neighbors(id).chain(self.in_neighbors(id))
    }

    fn out_neighbors(&self, id: &Self::NodeId) -> impl Iterator<Item = &Self::NodeId> {
        self.edges
            .iter()
            .filter_map(move |((from, to), _)| if from == id { Some(to) } else { None })
    }

    fn in_neighbors(&self, id: &Self::NodeId) -> impl Iterator<Item = &Self::NodeId> {
        self.edges
            .iter()
            .filter_map(move |((from, to), _)| if to == id { Some(from) } else { None })
    }

    fn all_node_ids(&self) -> Vec<Self::NodeId> {
        self.nodes
            .iter()
            .map(|(node_id, _)| node_id.clone())
            .collect()
    }
}

impl GraphAlgorithms for Graph {
    fn topological_sort(&self) -> Result<Vec<&Self::NodeId>, Self::Error> {
        let all_nodes: Vec<&Self::NodeId> = self.nodes.keys().collect();
        let node_count = all_nodes.len();

        let mut adj: HashMap<Self::NodeId, Vec<Self::NodeId>> = HashMap::new();
        let mut in_degree: HashMap<Self::NodeId, usize> = HashMap::new();

        for &node in &all_nodes {
            in_degree.insert(node.clone(), 0);
        }

        for (from, to) in self.edges.keys() {
            adj.entry(from.clone()).or_default().push(to.clone());
            *in_degree.entry(to.clone()).or_insert(0) += 1;
        }

        let mut queue: VecDeque<Self::NodeId> = in_degree
            .iter()
            .filter_map(|(node, &deg)| if deg == 0 { Some(node.clone()) } else { None })
            .collect();

        let mut order = Vec::with_capacity(node_count);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            if let Some(neighbors) = adj.get(&node) {
                for neighbor in neighbors {
                    let deg = in_degree.get_mut(neighbor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        if order.len() != node_count {
            return Err(GraphError::CycleDetected);
        }

        let result_refs: Vec<&Self::NodeId> = order
            .iter()
            .map(|id| self.nodes.get_key_value(id).unwrap().0)
            .collect();

        Ok(result_refs)
    }

    fn has_self_loops(&self) -> bool {
        self.edges().any(|(from, to, _)| from == to)
    }

    fn is_simple(&self) -> bool {
        if self.has_self_loops() {
            return false;
        }

        let mut seen = HashSet::new();
        for (from, to, _) in self.edges() {
            if !seen.insert((from, to)) {
                return false;
            }
        }
        true
    }

    fn has_cycle(&self) -> bool {
        let mut state: HashMap<Self::NodeId, u8> = HashMap::new();

        fn dfs<G: AdvancedAccess>(
            graph: &G,
            node_id: &G::NodeId,
            state: &mut HashMap<G::NodeId, u8>,
        ) -> bool {
            *state.entry(node_id.clone()).or_insert(0) = 1;
            for neighbor in graph.out_neighbors(node_id) {
                let st = *state.get(neighbor).unwrap_or(&0);
                if st == 1 {
                    return true;
                }
                if st == 0 && dfs(graph, neighbor, state) {
                    return true;
                }
            }

            false
        }

        for (node_id, _) in self.nodes() {
            if !state.contains_key(node_id) && dfs(self, node_id, &mut state) {
                return true;
            }
        }
        false
    }

    fn is_dag(&self) -> bool {
        !self.has_cycle()
    }

    fn is_connected(&self) -> bool {
        let all_nodes = self.all_node_ids();
        if all_nodes.is_empty() {
            return true;
        }
        let mut visited = HashSet::new();
        let mut stack = vec![all_nodes[0]];

        while let Some(node_id) = stack.pop() {
            if !visited.insert(node_id) {
                continue;
            }
            for nb in self.neighbors(&node_id) {
                if !visited.contains(nb) {
                    stack.push(nb.clone());
                }
            }
        }

        visited.len() == all_nodes.len()
    }

    fn is_bipartite(&self) -> bool {
        let mut color = HashMap::new();
        for (start, _) in self.nodes() {
            if color.contains_key(start) {
                continue;
            }
            let mut queue = VecDeque::new();
            color.insert(start.clone(), true);
            queue.push_back(start.clone());
            while let Some(node) = queue.pop_front() {
                let cur = color[&node];
                for neighbor in self.neighbors(&node) {
                    match color.get(neighbor) {
                        Some(&c) => {
                            if c == cur {
                                return false;
                            }
                        }
                        None => {
                            color.insert(neighbor.clone(), !cur);
                            queue.push_back(neighbor.clone());
                        }
                    }
                }
            }
        }
        true
    }

    fn is_strongly_connect(&self) -> bool {
        let nodes = self.all_node_ids();
        if nodes.is_empty() {
            return true;
        }

        let start = nodes[0];
        let reachable_from_start = self.reachable_from(&start);
        if reachable_from_start.len() != nodes.len() {
            return false;
        }
        let rev_reachable = self.reverse_reachable_from(&start);
        rev_reachable.len() == nodes.len()
    }

    fn reachable_from(&self, start: &Self::NodeId) -> HashSet<Self::NodeId> {
        let mut visited = HashSet::new();
        let mut stack = vec![start];

        while let Some(node_id) = stack.pop() {
            if !visited.insert(node_id.clone()) {
                continue;
            }
            for nb in self.out_neighbors(node_id) {
                if !visited.contains(nb) {
                    stack.push(nb);
                }
            }
        }

        visited
    }

    fn reverse_reachable_from(&self, start: &Self::NodeId) -> HashSet<Self::NodeId> {
        let mut visited = HashSet::new();
        let mut stack = vec![start];

        while let Some(node_id) = stack.pop() {
            if !visited.insert(node_id.clone()) {
                continue;
            }
            for nb in self.in_neighbors(node_id) {
                if !visited.contains(nb) {
                    stack.push(nb);
                }
            }
        }

        visited
    }

    /// Kosaraju
    fn strongly_connected_components(&self) -> Vec<Vec<Self::NodeId>> {
        let nodes = self.all_node_ids();
        let mut visited = HashSet::new();
        let mut order = Vec::new();

        fn dfs1<G: AdvancedAccess>(
            graph: &G,
            node_id: &G::NodeId,
            visited: &mut HashSet<G::NodeId>,
            order: &mut Vec<G::NodeId>,
        ) {
            if !visited.insert(*node_id) {
                return;
            }
            for nb in graph.out_neighbors(node_id) {
                dfs1(graph, nb, visited, order);
            }
            order.push(*node_id);
        }

        for node_id in nodes {
            if !visited.contains(&node_id) {
                dfs1(self, &node_id, &mut visited, &mut order);
            }
        }

        let mut visited_rev = HashSet::new();
        let mut components = Vec::new();

        fn dfs2<G: AdvancedAccess>(
            graph: &G,
            node_id: &G::NodeId,
            visited: &mut HashSet<G::NodeId>,
            components: &mut Vec<G::NodeId>,
        ) {
            if !visited.insert(*node_id) {
                return;
            }
            components.push(*node_id);
            for nb in graph.out_neighbors(node_id) {
                dfs2(graph, nb, visited, components);
            }
        }

        for node_id in order.into_iter().rev() {
            if !visited_rev.contains(&node_id) {
                let mut comp = Vec::new();
                dfs2(self, &node_id, &mut visited_rev, &mut comp);
                components.push(comp);
            }
        }

        components
    }

    fn shorted_path_unweighted(
        &self,
        from: &Self::NodeId,
        to: &Self::NodeId,
    ) -> Option<Vec<Self::NodeId>> {
        if from == to {
            return Some(vec![from.clone()]);
        }

        let mut queue = VecDeque::new();
        let mut parent = HashMap::new();

        queue.push_back(from.clone());
        parent.insert(from.clone(), None);

        while let Some(node_id) = queue.pop_front() {
            for nb in self.out_neighbors(&node_id) {
                if !parent.contains_key(nb) {
                    parent.insert(nb.clone(), Some(node_id));
                    if nb == to {
                        let mut path = Vec::new();
                        let mut cur = nb.clone();
                        while let Some(prev) = parent[&cur] {
                            path.push(cur.clone());
                            cur = prev.clone();
                        }
                        path.push(from.clone());
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(nb.clone());
                }
            }
        }

        None
    }

    fn degree_distribution(&self) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();
        for (node_id, _) in self.nodes() {
            let deg = self.out_neighbors(node_id).count() + self.in_neighbors(node_id).count();
            *distribution.entry(deg).or_insert(0) += 1;
        }
        distribution
    }

    fn transitive_closure(&self) -> HashMap<Self::NodeId, Vec<Self::NodeId>> {
        let nodes = self.all_node_ids();
        let index: HashMap<_, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let n = nodes.len();
        let mut reach = vec![vec![false; n]; n];
        for (from, to, _) in self.edges() {
            let i = index[from];
            let j = index[to];
            reach[i][j] = true;
        }
        for i in 0..n {
            reach[i][i] = true;
        }
        for k in 0..n {
            for i in 0..n {
                if reach[i][k] {
                    for j in 0..n {
                        if reach[k][j] {
                            reach[i][j] = true;
                        }
                    }
                }
            }
        }
        let mut res = HashMap::new();
        for (i, node) in nodes.iter().enumerate() {
            let reachable: Vec<_> = (0..n)
                .filter(|&j| reach[i][j])
                .map(|j| nodes[j].clone())
                .collect();
            res.insert(node.clone(), reachable);
        }
        res
    }
}
