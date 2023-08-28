
// mod optimizer;
mod utils;
use debug::PrintTrait;
use traits::Into;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::core::{Tensor, TensorTrait, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};
use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::fp8x23::core::{FP8x23Add, FP8x23Div, FP8x23Mul, FP8x23Sub, FP8x23Impl};
use utils::optimizer_utils::{exponential_weights, diagonalize};

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
    
    // Test exponential weights
    // let mut ex = exponential_weights(97, 5);
    // let mut i = 0;
    // loop {
    //     if i == 5 {
    //         break ();
    //     }
    //     let mut ex_i: u32 = *ex.data.at(i).mag;
    //     ex_i.print();
    //     i += 1;
    // };

    // Test diagonalize
    let mut shape1 = ArrayTrait::<u32>::new();
    shape1.append(2);
    let mut data1 = ArrayTrait::<FixedType>::new();
    data1.append(FixedTrait::new_unscaled(1, false));
    data1.append(FixedTrait::new_unscaled(2, false));
    let mut diag_test = TensorTrait::<FixedType>::new(shape1.span(), data1.span(), extra);
    let mut diag_out = diagonalize(diag_test);
    let mut i = 0;
    loop {
        if i == 2 {
            break ();
        }
        let mut diag_out_i: u32 = *diag_out.data.at(i).mag;
        diag_out_i.print();
        i += 1;
    }

}