# Copyright (c) 2026 ARC (Applied Research & Computation)
# SPDX-License-Identifier: LGPL-2.1-or-later

from substrate_core import Array


def test_array_benchmark(benchmark):
    def create():
        arr = Array(5)
        return arr.length()

    result = benchmark(create)
    assert result == 5
