// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::equation::expression::Equation;

use crate::Expr;

impl Equation for Expr {
    fn lhs_variable(&self) -> Option<String> {
        // For simplicity, assume an equation is stored as "Var = Expr"
        // Here we treat the expression itself as the RHS; LHS is stored separately.
        // We'll implement `with_lhs` to wrap the expression.
        None // default – no LHS stored
    }

    fn with_lhs(self, _var: &str) -> Self {
        // We can store LHS as metadata, but for evaluation we ignore LHS.
        // A proper implementation would wrap the equation in a struct { lhs: String, rhs: Expr }
        // For now, we just return self (the RHS) and store LHS elsewhere.
        self
    }
}
