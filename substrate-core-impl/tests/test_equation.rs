// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::collections::HashMap;

use substrate_core_impl::Expr;
use substrate_core_spec::equation::expression::{Differentiable, Expression, ExpressionBuilder};

#[test]
fn test_equation_creation() {
    /*
    let x = Expr::variable("x");
    let y = Expr::variable("y");
    let expr = x.clone().add(y.clone()).mul(Expr::constant(2.0)); // 2*(x+y)

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);
    vars.insert("y".to_string(), 5.0);
    assert_eq!(expr.eval(&vars).unwrap(), 16.0);

    let dx = expr.derivative("x").unwrap();
    assert_eq!(dx.eval(&vars).unwrap(), 2.0);
    */
}
