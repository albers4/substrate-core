use substrate_core_impl::DenseArray;

#[test]
fn test_dense_array_capacity() {
    let arr: DenseArray<f64> = DenseArray::new(10_000);
    assert_eq!(arr.capacity(), 10_000);
}
