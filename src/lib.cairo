
// mod optimizer;
mod utils;
use debug::PrintTrait;
use traits::Into;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::core::{Tensor, TensorTrait, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
use utils::optimizer_utils::{exponential_weights};

#[test]
#[available_gas(99999999999999999)]
fn test() {
    // Build test 5x2 matrix with randomly generated values
    let mut shape = ArrayTrait::<u32>::new();
    shape.append(2);
    shape.append(5);

    let mut data = ArrayTrait::<u32>::new();
    data.append(0);
    data.append(46);
    data.append(57);
    data.append(89);
    data.append(21);
    data.append(63);
    data.append(59);
    data.append(87);
    data.append(61);
    data.append(10);

    let extra = Option::<ExtraParams>::None(());

    let mut X_test = TensorTrait::<u32>::new(shape.span(), data.span(), extra);
    // let mut X_test_rolling_cov = rolling_covariance(X_test, 97, 4);
    // let mut X_test_cov_1 = *X_test_rolling_cov.at(0);

    // assert(*X_test_cov_1.data.at(0) - FixedTrait::new_unscaled(527, false) < 1);
    let mut ex = exponential_weights(97, 5);
    let mut i = 0;
    loop {
        if i == 5 {
            break ();
        }
        let mut ex_i: u32 = *ex.data.at(i).mag;
        ex_i.print();
        i += 1;
    };
}