// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::collections::HashMap;

pub trait Expression: Sized {
    type Value;
    type Error;

    fn eval(&self, vars: &HashMap<String, Self::Value>) -> Result<Self::Value, Self::Error>;
    fn variables(&self) -> Vec<String>;
    fn is_constant(&self) -> bool;
    fn substitute(&self, var_map: &HashMap<String, Self>) -> Result<Self, Self::Error>;
}

pub trait ExpressionBuilder: Expression {
    fn constant(value: Self::Value) -> Self;
    fn variable(name: &str) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn neg(self) -> Self;
    fn pow(self, exponent: Self) -> Self;

    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
}

pub trait Differentiable: Expression {
    fn derivative(&self, var: &str) -> Result<Self, Self::Error>;
    fn gradient(&self, vars: &[String]) -> Result<Vec<Self>, Self::Error>;
}

pub trait Equation: Expression {
    fn lhs_variable(&self) -> Option<String>;
    fn with_lhs(self, var: &str) -> Self;
}

pub trait EquationSystemLike: Sized {
    type Expr: Equation;
    type Error;

    fn add_equation(&mut self, eq: Self::Expr) -> Result<(), Self::Error>;
    fn equations(&self) -> &[Self::Expr];
    fn evaluate_all(
        &self,
        vars: &HashMap<String, <Self::Expr as Expression>::Value>,
    ) -> Result<HashMap<String, <Self::Expr as Expression>::Value>, Self::Error>;
    fn dependency_order(&self) -> Result<Vec<usize>, Self::Error>;
}
