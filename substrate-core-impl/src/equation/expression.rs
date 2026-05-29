// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::fmt::Display;

use substrate_core_spec::equation::expression::Expression;

use crate::equation::error::ExpressionError;

#[derive(Debug, Clone)]
pub enum Expr {
    Const(f64),
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Exp(Box<Expr>),
    Ln(Box<Expr>),
}

impl Expr {
    fn collect_vars(&self, out: &mut Vec<String>) {
        match self {
            Expr::Var(name) => out.push(name.clone()),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                a.collect_vars(out);
                b.collect_vars(out);
            }
            Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a) | Expr::Ln(a) => {
                a.collect_vars(out);
            }
            Expr::Pow(a, b) => {
                a.collect_vars(out);
                b.collect_vars(out);
            }
            Expr::Const(_) => {}
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Const(c) => write!(f, "{}", c),
            Expr::Var(name) => write!(f, "{}", name),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Neg(a) => write!(f, "-({})", a),
            Expr::Pow(a, b) => write!(f, "({} ^ {})", a, b),
            Expr::Sin(a) => write!(f, "sin({})", a),
            Expr::Cos(a) => write!(f, "cos({})", a),
            Expr::Exp(a) => write!(f, "exp^({})", a),
            Expr::Ln(a) => write!(f, "ln({})", a),
        }
    }
}

impl Expression for Expr {
    type Value = f64;
    type Error = ExpressionError;

    fn eval(
        &self,
        vars: &std::collections::HashMap<String, Self::Value>,
    ) -> Result<Self::Value, Self::Error> {
        match self {
            Expr::Const(c) => Ok(*c),
            Expr::Var(name) => vars
                .get(name)
                .copied()
                .ok_or(ExpressionError::VariableNotFound(name.to_string())),
            Expr::Add(a, b) => Ok(a.eval(vars)? + b.eval(vars)?),
            Expr::Sub(a, b) => Ok(a.eval(vars)? - b.eval(vars)?),
            Expr::Mul(a, b) => Ok(a.eval(vars)? * b.eval(vars)?),
            Expr::Div(a, b) => {
                let denom = b.eval(vars)?;
                if denom == 0.0 {
                    Err(ExpressionError::DivisionByZero)
                } else {
                    Ok(a.eval(vars)? / denom)
                }
            }
            Expr::Neg(a) => Ok(-a.eval(vars)?),
            Expr::Pow(a, b) => Ok(a.eval(vars)?.powf(b.eval(vars)?)),
            Expr::Sin(a) => Ok(a.eval(vars)?.sin()),
            Expr::Cos(a) => Ok(a.eval(vars)?.cos()),
            Expr::Exp(a) => Ok(a.eval(vars)?.exp()),
            Expr::Ln(a) => {
                let val = a.eval(vars)?;
                if val <= 0.0 {
                    Err(ExpressionError::DomainError("ln of non-positive".into()))
                } else {
                    Ok(val.ln())
                }
            }
        }
    }

    fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_vars(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn is_constant(&self) -> bool {
        match self {
            Expr::Const(_) => true,
            Expr::Var(_) => false,
            Expr::Add(a, b) => a.is_constant() && b.is_constant(),
            Expr::Sub(a, b) => a.is_constant() && b.is_constant(),
            Expr::Mul(a, b) => a.is_constant() && b.is_constant(),
            Expr::Div(a, b) => a.is_constant() && b.is_constant(),
            Expr::Neg(a) => a.is_constant(),
            Expr::Pow(a, b) => a.is_constant() && b.is_constant(),
            Expr::Sin(a) => a.is_constant(),
            Expr::Cos(a) => a.is_constant(),
            Expr::Exp(a) => a.is_constant(),
            Expr::Ln(a) => a.is_constant(),
        }
    }

    fn substitute(
        &self,
        var_map: &std::collections::HashMap<String, Self>,
    ) -> Result<Self, Self::Error> {
        match self {
            Expr::Const(c) => Ok(Expr::Const(*c)),
            Expr::Var(name) => Ok(var_map
                .get(name)
                .cloned()
                .unwrap_or_else(|| Expr::Var(name.clone()))),
            Expr::Add(a, b) => Ok(Expr::Add(
                Box::new(a.substitute(var_map)?),
                Box::new(b.substitute(var_map)?),
            )),
            Expr::Sub(a, b) => Ok(Expr::Sub(
                Box::new(a.substitute(var_map)?),
                Box::new(b.substitute(var_map)?),
            )),
            Expr::Mul(a, b) => Ok(Expr::Mul(
                Box::new(a.substitute(var_map)?),
                Box::new(b.substitute(var_map)?),
            )),
            Expr::Div(a, b) => Ok(Expr::Div(
                Box::new(a.substitute(var_map)?),
                Box::new(b.substitute(var_map)?),
            )),
            Expr::Neg(a) => Ok(Expr::Neg(Box::new(a.substitute(var_map)?))),
            Expr::Pow(a, b) => Ok(Expr::Pow(
                Box::new(a.substitute(var_map)?),
                Box::new(b.substitute(var_map)?),
            )),
            Expr::Sin(a) => Ok(Expr::Sin(Box::new(a.substitute(var_map)?))),
            Expr::Cos(a) => Ok(Expr::Cos(Box::new(a.substitute(var_map)?))),
            Expr::Exp(a) => Ok(Expr::Exp(Box::new(a.substitute(var_map)?))),
            Expr::Ln(a) => Ok(Expr::Ln(Box::new(a.substitute(var_map)?))),
        }
    }
}
