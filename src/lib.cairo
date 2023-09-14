
// mod optimizer;
mod utils;
use debug::PrintTrait;
use option::OptionTrait;
use array::{ArrayTrait, SpanTrait};
use traits::{Into, TryInto};
use dict::Felt252DictTrait;
use orion::operators::tensor::{
    core::{Tensor, TensorTrait, ExtraParams},
    implementations::{
        impl_tensor_u32::{Tensor_u32},
        impl_tensor_fp::{Tensor_fp, FixedTypeTensorMul, FixedTypeTensorSub, FixedTypeTensorDiv}
    },
    math::arithmetic::arithmetic_fp::core::{add, sub, mul, div}
};
use orion::numbers::fixed_point::{
    core::{FixedTrait, FixedType, FixedImpl},
    // implementations::fp8x23::core::{FP8x23Add, FP8x23Div, FP8x23Mul, FP8x23Sub, FP8x23Impl},
    implementations::fp16x16::core::{FP16x16Add, FP16x16Div, FP16x16Mul, FP16x16Sub, FP16x16Impl},
};
use alexandria_data_structures::vec::{VecTrait, Felt252VecImpl, Felt252Vec};
use utils::optimizer_utils::{exponential_weights, 
                             diagonalize, 
                             weighted_covariance, 
                             rolling_covariance,
                             Matrix,
                             forward_elimination,
                             back_substitution,
                             linalg_solve,
                             test_tensor};

#[test]
#[available_gas(99999999999999999)]
fn test<impl TDrop: Drop<FixedType>,
        impl FixedDict: Felt252DictTrait<FixedType>,
        impl Vec: VecTrait<Felt252Vec<FixedType>, usize>,
        impl VecDrop: Drop<Felt252Vec<FixedType>>,
        impl VecCopy: Copy<Felt252Vec<FixedType>>>() {
    
    let extra = ExtraParams {fixed_point: Option::Some(FixedImpl::FP16x16(()))};

    // Build test 5x2 matrix with randomly generated values
    // let mut shape = ArrayTrait::<u32>::new();
    // shape.append(5);
    // shape.append(2);

    // let mut data = ArrayTrait::<FixedType>::new();
    // data.append(FixedTrait::new_unscaled(0, false));
    // data.append(FixedTrait::new_unscaled(46, false));
    // data.append(FixedTrait::new_unscaled(57, false));
    // data.append(FixedTrait::new_unscaled(89, false));
    // data.append(FixedTrait::new_unscaled(21, false));
    // data.append(FixedTrait::new_unscaled(63, false));
    // data.append(FixedTrait::new_unscaled(59, false));
    // data.append(FixedTrait::new_unscaled(87, false));
    // data.append(FixedTrait::new_unscaled(61, false));
    // data.append(FixedTrait::new_unscaled(10, false));
    // let mut X_test = TensorTrait::<FixedType>::new(shape.span(), data.span(), Option::Some(extra));
    
    // // Test exponential weights
    // let mut weights = exponential_weights(97, 5);
    // let mut i = 0;
    // loop {
    //     if i == 2 {
    //         break ();
    //     }
    //     let mut ex_i: u32 = *weights.data.at(i).mag;
    //     ex_i.print();
    //     i += 1;
    // };

    // // Test diagonalize
    // let mut shape1 = ArrayTrait::<u32>::new();
    // shape1.append(2);
    // let mut data1 = ArrayTrait::<FixedType>::new();
    // data1.append(FixedTrait::new_unscaled(1, false));
    // data1.append(FixedTrait::new_unscaled(2, false));
    // let mut diag_test = TensorTrait::<FixedType>::new(shape1.span(), data1.span(), extra);
    // let mut diag_out = diagonalize(diag_test);
    // let mut i = 0;
    // loop {
    //     if i == 2 {
    //         break ();
    //     }
    //     let mut diag_out_i: u32 = *diag_out.data.at(i).mag;
    //     diag_out_i.print();
    //     i += 1;
    // };

    // Test covariance
    // let cov_X = weighted_covariance(X_test, weights);
    // let mut i = 0;
    // loop {
    //     if i == 4 {
    //         break ();
    //     }
    //     let mut cov_X_i = *cov_X.data.at(i).mag;
    //     cov_X_i.print();
    //     i += 1;
    // };

    // Test rolling covariance
    // let rolling = rolling_covariance(X_test, 97, 3);
    // let mut i = 0;
    // loop {
    //     i.print();
    //     if i == (rolling.len() - 1) {
    //         break ();
    //     }
    //     let cov_i = *rolling.at(i);
    //     let mut j = 0;
    //     loop {
    //         if j == (*cov_i.shape.at(0) * *cov_i.shape.at(1)) {
    //             break ();
    //         }
    //         let mut cov_i_j = *cov_i.data.at(j).mag;
    //         cov_i_j.print();
    //         j += 1;
    //     };
    //     i += 1;
    // };

    // Test forward elimination
    // Build test 3x3 matrix
    // TODO: Fix this so that we can test the new functions...
    
    let mut X_data_items: Felt252Dict<FixedType> = Default::default();
    X_data_items.insert(0, FixedTrait::new_unscaled(2, false));
    X_data_items.insert(1, FixedTrait::new_unscaled(1, false));
    X_data_items.insert(2, FixedTrait::new_unscaled(1, true));
    X_data_items.insert(3, FixedTrait::new_unscaled(3, true));
    X_data_items.insert(4, FixedTrait::new_unscaled(1, true));
    X_data_items.insert(5, FixedTrait::new_unscaled(2, false));
    X_data_items.insert(6, FixedTrait::new_unscaled(2, true));
    X_data_items.insert(7, FixedTrait::new_unscaled(1, false));
    X_data_items.insert(8, FixedTrait::new_unscaled(2, false));
    let mut X_data: Felt252Vec<FixedType> = Felt252Vec {items: X_data_items, len: 3};
    let mut X = Matrix {rows: 3, 
                        cols: 3, 
                        data: X_data};
    
    let mut y_items: Felt252Dict<FixedType> = Default::default();
    y_items.insert(0, FixedTrait::new_unscaled(8, false));
    y_items.insert(1, FixedTrait::new_unscaled(11, true));
    y_items.insert(2, FixedTrait::new_unscaled(3, true));
    let mut y: Felt252Vec<FixedType> = Felt252Vec {items: y_items, len: 3};
    
    let mut sol = linalg_solve(ref X, ref y, 3);
    test_tensor(sol);
    
}