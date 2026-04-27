from substrate_core import DenseArray


def test_dense_array_capacity_benchmark(benchmark):
    def create_and_capacity():
        arr = DenseArray(10_000)
        return arr.capacity()

    result = benchmark(create_and_capacity)
    assert result == 10_000
