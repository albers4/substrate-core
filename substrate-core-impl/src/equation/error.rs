// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#[derive(Debug)]
pub enum ExpressionError {
    VariableNotFound(String),
    DivisionByZero,
    DomainError(String),
    DifferentiationError(String),
}

#[derive(Debug)]
pub enum EquationError {
    FailedToEvaluate,
}
