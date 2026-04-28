// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub enum MemoryOrder {
    #[default]
    RowMajor,
    ColumnMajor,
}
