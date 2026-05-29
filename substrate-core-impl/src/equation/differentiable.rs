// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::equation::expression::{Differentiable, ExpressionBuilder};

use crate::Expr;

impl Differentiable for Expr {
    fn derivative(&self, var: &str) -> Result<Self, Self::Error> {
        match self {
            Expr::Const(_) => Ok(Expr::Const(0.0)),
            Expr::Var(name) => {
                if name == var { Ok(Expr::Const(1.0)) }
                else { Ok(Expr::Const(0.0)) }
            },
            Expr::Add(a, b) => Ok(a.derivative(var)?.add(b.derivative(var)?)),
            Expr::Sub(a, b) => Ok(a.derivative(var)?.sub(b.derivative(var)?)),
            Expr::Mul(a, b) => {
                let a_prime = a.derivative(var)?;
                let b_prime = b.derivative(var)?;
                Ok(a_prime.mul(*b.clone()).add(a.clone().mul(b_prime)))
            },
            Expr::Div(a, b) => {
                let a_prime = a.derivative(var)?;
                let b_prime = b.derivative(var)?;
                let numerator = a_prime.mul(*b.clone()).add(a.clone().mul(b_prime));
                let denominator = b.clone().mul(*b.clone());
                Ok(numerator.div(denominator))
            },
            Expr::Neg(a) => Ok(a.derivative(var)?.neg()),
            Expr::Pow(a, b) => {
                let u = a.clone();
                let v = b.clone();
                let u_prime = u.derivative(var)?;
                let v_prime = v.derivative(var)?;
                let term1 = v_prime.mul(*u.clone()).ln();
                let term2 = v.clone().mul(u_prime.div(*u.clone()));
                let pow_expr = u.pow(*v);

                Ok(pow_expr.mul(term1.add(term2)))
            },
            Expr::Sin(a) => Ok(a.derivative(var)?.mul(a.clone().cos())),
            Expr::Cos(a) => Ok(a.derivative(var)?.mul(a.clone().sin().neg())),
            Expr::Exp(a) => Ok(a.derivative(var)?.mul(self.clone())),
            Expr::Ln(a) => Ok(a.derivative(var)?.div(*a.clone())),
        }
    }

    fn gradient(&self, vars: &[String]) -> Result<Vec<Self>, Self::Error> {
        vars.iter().map(|v| self.derivative(v)).collect()
    }
}