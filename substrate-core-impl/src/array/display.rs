use core::any::type_name;
use core::fmt::Display;

use substrate_core_spec::array::ArrayLike;

use crate::Array;
use crate::array::ArrayView;

impl Display for Array<f64, Vec<f64>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Array<shape={:?}, dtype={}>",
            self.shape(),
            type_name::<f64>()
        )
    }
}

impl<'a> Display for ArrayView<'a, f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ArrayView<shape={:?}, dtype={}>",
            self.shape(),
            type_name::<f64>()
        )
    }
}
