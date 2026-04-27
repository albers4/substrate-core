from substrate_core import DenseArray


def test_dense_array_len():
    arr = DenseArray(10)
    assert arr.capacity() == 10
