# Copyright (c) 2026 ARC (Applied Research & Computation)
# SPDX-License-Identifier: LGPL-2.1-or-later

from substrate_core import Array


def test_array_len():
    arr = Array(5)
    assert arr.length() == 5
