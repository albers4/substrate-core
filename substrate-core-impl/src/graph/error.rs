// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#[derive(std::fmt::Debug)]
pub enum GraphError {
    NotImplemented,
    NotAnArray,
    FailedToGetNode,
    NodeInputInvalid,
    CycleDetected,
}

#[derive(std::fmt::Debug)]
pub enum NodeError {
    NotImplemented,
}
