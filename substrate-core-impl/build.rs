// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::env;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .compiler("clang++-16")
        .flag("-std=c++17")
        .flag("-O3") 
        .flag("-fno-exceptions")
        .flag("-march=native")
        .include("cpp")
        .file("cpp/array_ops.cpp");

    if let Ok(enzyme_path) = env::var("ENZYME_PATH") {
        build.flag("-Xclang").flag("-load").flag("-Xclang").flag(&enzyme_path);
        build.flag(&format!("-fpass-plugin={}", enzyme_path));
    } else {
        panic!("ENZYME_PATH not set. Enzyme is required for this build.");
    }

    build.out_dir(&out_dir).compile("array_ops");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=array_ops");
    println!("cargo:rerun-if-changed=cpp/");
}