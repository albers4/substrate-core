// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#[derive(Debug)]
pub enum ArrayError {
    ArrayFromLinspaceError,
    IndexOutOfBounds,
    IndexConversionError,
    EmptyArray,
    ArrayNotAScalar,
    DimensionMismatch,
    EmptyShape,
    InvalidShapeDimension,
    ReshapeSizeMismatch,
    NotContiguous,
    ValidForMatricesOnly,
    AxisOutOfBounds,
    InvalidSlice,
    IncompatibleShapes,
    InvalidAxisLength,
    InvalidStep,
    InvalidParameter,
    InvalidSplit,
    NotImplemented,
    ValidForVectorsOnly,
}
