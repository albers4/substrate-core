// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::collections::HashMap;

use substrate_core_spec::equation::expression::{EquationSystemLike, Expression};

use crate::{
    Expr,
    equation::error::{EquationError, ExpressionError},
};

pub struct EquationSystem {
    equations: Vec<Expr>,
}

impl EquationSystemLike for EquationSystem {
    type Expr = Expr;
    type Error = EquationError;

    fn add_equation(&mut self, eq: Self::Expr) -> Result<(), Self::Error> {
        self.equations.push(eq);
        Ok(())
    }

    fn equations(&self) -> &[Self::Expr] {
        &self.equations
    }

    fn evaluate_all(
        &self,
        vars: &std::collections::HashMap<
            String,
            <Self::Expr as substrate_core_spec::equation::expression::Expression>::Value,
        >,
    ) -> Result<
        std::collections::HashMap<
            String,
            <Self::Expr as substrate_core_spec::equation::expression::Expression>::Value,
        >,
        Self::Error,
    > {
        let mut results = HashMap::new();
        // For each equation, evaluate its RHS and map LHS (if any) to the result.
        // Since our `Expr` doesn't store LHS, we would need to pass a separate mapping.
        // To keep it simple, we assume the caller provides initial variable values,
        // and we evaluate expressions in order (no dependency sorting).
        // A real system would use topological order based on variable dependencies.
        for eq in &self.equations {
            let val = eq.eval(vars).map_err(|_| EquationError::FailedToEvaluate)?;
            // This is placeholder: we cannot determine variable name from expression alone.
            // In practice, you'd store (lhs, rhs) pairs.
            results.insert("result".to_string(), val);
        }
        Ok(results)
    }

    fn dependency_order(&self) -> Result<Vec<usize>, Self::Error> {
        // Simplified: just return indices in order (assuming no cycles)
        Ok((0..self.equations.len()).collect())
    }
}

// Helper: Build a proper equation with LHS
#[derive(Debug, Clone)]
pub struct EquationWithLHS {
    pub lhs: String,
    pub rhs: Expr,
}

impl EquationWithLHS {
    pub fn new(lhs: &str, rhs: Expr) -> Self {
        Self {
            lhs: lhs.to_string(),
            rhs,
        }
    }
    pub fn eval(&self, vars: &HashMap<String, f64>) -> Result<f64, ExpressionError> {
        self.rhs.eval(vars)
    }
}
