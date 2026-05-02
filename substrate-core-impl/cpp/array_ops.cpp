// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "array_ops.h"
#include <vector>
#include <cstring>
#include <cstdio>

extern "C" int enzyme_dup;
extern "C" int enzyme_const;
extern "C" void __enzyme_autodiff(void*, ...);
extern "C" void __enzyme_fwddiff(void*, ...);

struct ArrayHandle {
    std::vector<double> data;
    size_t len;

    ArrayHandle(const double* d, size_t length)
        : data(d, d + length), len(length) {}
};

extern "C" ArrayHandle* array_create(const double* data, size_t len) {
    auto *arr = new ArrayHandle(data, len);
    arr->len = len;
    return arr;
}

extern "C" void array_add(ArrayHandle *res, const ArrayHandle *a, const ArrayHandle *b) {
    res->len = a->len;

    for (size_t i = 0; i < res->len; ++i) {
        res->data[i] = a->data[i] + b->data[i];
    }
}

// Reverse mode (Adjoint)
extern "C" void array_add_backward(
    ArrayHandle* res, ArrayHandle* dres,
    ArrayHandle* a,   ArrayHandle* da,
    ArrayHandle* b,   ArrayHandle* db
) {
    __enzyme_autodiff((void*) array_add,
                      enzyme_dup, res, dres,
                      enzyme_dup, a, da,
                      enzyme_dup, b, db);
}

// Forward mode (Tangent)
extern "C" void array_add_forward(
    ArrayHandle* res, ArrayHandle* dres,
    ArrayHandle* a,   ArrayHandle* da,
    ArrayHandle* b,   ArrayHandle* db
) {
    __enzyme_fwddiff((void*) array_add,
                     enzyme_dup, res, dres,
                     enzyme_dup, a, da,
                     enzyme_dup, b, db);
}

extern "C" void array_copy(const ArrayHandle *arr, double *buffer, size_t len) {
    memcpy(buffer, arr->data.data(), len * sizeof(double));
}

extern "C" void array_destroy(ArrayHandle *arr) {
    delete arr;
}