// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use substrate_core_spec::semantics::literal::LiteralUnit;

#[test]
fn test_literals() {
    let a = 1_f64.m();
    let b = 100_f64.cm();
    assert_eq!(a, b);
    let c = a + b;
    assert_eq!(c, 2_f64.m());

    let a = 2_f64.m();
    let b = 100_f64.cm();
    let c = a - b;
    assert_eq!(c, 1_f64.m());

    let a = 1_f64.m();
    let b = 100_f64.cm();
    let c = a * b;
    assert_eq!(c, 1_f64.m2());
}

#[test]
fn test_all_si() {
    let _a = 1_f64.m();
    let _b = 1_f64.kg();
    let _c = 1_f64.s();
    let _d = 1_f64.ampere();
    let _e = 1_f64.kelvin();
    let _f = 1_f64.mole();
    let _g = 1_f64.candela();
}

#[test]
fn test_temperature() {
    let _a = 20_f64.deg_c();
    let _b = 30_f64.deg_c();
}
