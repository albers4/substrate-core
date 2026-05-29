// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::equation::expression::ExpressionBuilder;

use crate::Expr;

impl ExpressionBuilder for Expr {
    fn constant(value: Self::Value) -> Self {
        Expr::Const(value)
    }

    fn variable(name: &str) -> Self {
        Expr::Var(name.to_string())
    }

    fn add(self, other: Self) -> Self {
        Expr::Add(Box::new(self), Box::new(other))
    }

    fn sub(self, other: Self) -> Self {
        Expr::Sub(Box::new(self), Box::new(other))
    }

    fn mul(self, other: Self) -> Self {
        Expr::Mul(Box::new(self), Box::new(other))
    }

    fn div(self, other: Self) -> Self {
        Expr::Div(Box::new(self), Box::new(other))
    }

    fn neg(self) -> Self {
        Expr::Neg(Box::new(self))
    }

    fn pow(self, exponent: Self) -> Self {
        Expr::Pow(Box::new(self), Box::new(exponent))
    }

    fn sin(self) -> Self {
        Expr::Sin(Box::new(self))
    }

    fn cos(self) -> Self {
        Expr::Cos(Box::new(self))
    }

    fn exp(self) -> Self {
        Expr::Exp(Box::new(self))
    }

    fn ln(self) -> Self {
        Expr::Ln(Box::new(self))
    }
}
