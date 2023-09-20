
// mod optimizer;
mod utils;
use debug::PrintTrait;
use option::OptionTrait;
use array::{ArrayTrait, SpanTrait};
use traits::{Into, TryInto};
use dict::Felt252DictTrait;
use nullable::{NullableTrait, nullable_from_box, match_nullable, FromNullableResult};
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, FP16x16TensorMul, FP16x16TensorSub, FP16x16TensorDiv};
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16, FP16x16Add, FP16x16Div, FP16x16Mul, FP16x16Sub, FP16x16Impl};
use alexandria_data_structures::vec::{VecTrait, Felt252VecImpl, Felt252Vec};
// use utils::optimizer_utils::{exponential_weights, 
//                              diagonalize, 
//                              weighted_covariance, 
//                              rolling_covariance,
//                              Matrix,
//                              forward_elimination,
//                              back_substitution,
//                              linalg_solve,
//                              test_tensor};

#[test]
#[available_gas(99999999999999999)]
fn test<impl TDrop: Drop<FP16x16>,
        impl FixedDict: Felt252DictTrait<FP16x16>,
        impl Vec: VecTrait<Felt252Vec<FP16x16>, usize>,
        impl VecDrop: Drop<Felt252Vec<FP16x16>>,
        impl VecCopy: Copy<Felt252Vec<FP16x16>>>() {
    

    // Build test 5x2 matrix with randomly generated values
    // let mut shape = ArrayTrait::<u32>::new();
    // shape.append(5);
    // shape.append(2);

    // let mut data = ArrayTrait::<FP16x16>::new();
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
    // let mut X_test = TensorTrait::<FP16x16>::new(shape.span(), data.span(), Option::Some(extra));
    
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
    // let mut data1 = ArrayTrait::<FP16x16>::new();
    // data1.append(FixedTrait::new_unscaled(1, false));
    // data1.append(FixedTrait::new_unscaled(2, false));
    // let mut diag_test = TensorTrait::<FP16x16>::new(shape1.span(), data1.span(), extra);
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
    
    // let mut X_data_items: Felt252Dict<Nullable<FP16x16>> = Default::default();

    // X_data_items.insert(0, NullableTrait::new(FP16x16 { mag: 2, sign: false }));
    // X_data_items.insert(1, NullableTrait::new(FP16x16 { mag: 1, sign: false }));
    // X_data_items.insert(2, NullableTrait::new(FP16x16 { mag: 1, sign: true }));
    // X_data_items.insert(3, NullableTrait::new(FP16x16 { mag: 3, sign: true }));
    // X_data_items.insert(4, NullableTrait::new(FP16x16 { mag: 1, sign: true }));
    // X_data_items.insert(5, NullableTrait::new(FP16x16 { mag: 2, sign: false }));
    // X_data_items.insert(6, NullableTrait::new(FP16x16 { mag: 2, sign: true }));
    // X_data_items.insert(7, NullableTrait::new(FP16x16 { mag: 1, sign: false }));
    // X_data_items.insert(8, NullableTrait::new(FP16x16 { mag: 2, sign: false })); 
    
    // let val = match match_nullable(val) {
    //     FromNullableResult::Null(()) => panic_with_felt252('No value found'),
    //     FromNullableResult::NotNull(val) => val.unbox(),
    // };



    // let mut X_data: Felt252Vec<Nullable<FP16x16>> = Felt252Vec {items: X_data_items, len: 3};
    // let mut X = Matrix {rows: 3, 
    //                     cols: 3, 
    //                     data: X_data};
    
    // let mut y_items: Felt252Dict<FP16x16> = Default::default();
    // y_items.insert(0, FixedTrait::new_unscaled(8, false));
    // y_items.insert(1, FixedTrait::new_unscaled(11, true));
    // y_items.insert(2, FixedTrait::new_unscaled(3, true));
    // let mut y: Felt252Vec<FP16x16> = Felt252Vec {items: y_items, len: 3};
    
    // let mut sol = linalg_solve(ref X, ref y);
    // test_tensor(sol);
    
}