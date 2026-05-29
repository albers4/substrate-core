// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

/*
use substrate_core_impl::{EdgeData, Graph, NodeData, NodeOperation, arr, ComputeGraph};
use substrate_core_spec::graph::ops::{AccessOps, AccessOpsMut};

#[test]
fn test_graph_array() {
    let mut graph = Graph::new();

    let a = graph.add_node(NodeData::array(arr!(1.0, 2.0, 3.0)));
    let b = graph.add_node(NodeData::array(arr!(4.0, 5.0, 6.0)));
    graph.add_edge(a, b, EdgeData::new()).unwrap();

    assert_eq!(graph.node_data(a), Some(&NodeData::array(arr!(1.0, 2.0, 3.0))));
}

#[test]
fn test_graph_operation() {
    let mut graph = Graph::new();

    let a = graph.add_node(NodeData::array(arr!(1.0, 2.0, 3.0)));
    let b = graph.add_node(NodeData::array(arr!(4.0, 5.0, 6.0)));
    let c = graph.add_node(NodeData::operation(NodeOperation::Add));

    graph.add_edge(a, c, EdgeData::new()).unwrap();
    graph.add_edge(b, c, EdgeData::new()).unwrap();

    let res = graph.evaluate(c).unwrap();
    assert_eq!(graph.get_array(c).unwrap(), &arr!(5.0, 7.0, 9.0));
}
*/
