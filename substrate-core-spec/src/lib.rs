/// Specification: v1.0
pub trait ArrayLike {
    type Item;
    type Error;

    fn get(&self, index: usize) -> Result<&Self::Item, Self::Error>;
}

pub trait ArrayLikeMut: ArrayLike {
    fn set(&mut self, index: usize, value: Self::Item) -> Result<(), Self::Error>;
}
