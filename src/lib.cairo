
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
use alexandria_data_structures::vec::{Felt252Vec, NullableVecImpl, NullableVec, VecTrait};
use utils::optimizer_utils::{Matrix,
                             forward_elimination,
                             back_substitution,
                             linalg_solve,
                             test_tensor};

#[test]
#[available_gas(99999999999999999)]
fn test<impl FixedDict: Felt252DictTrait<FP16x16>, impl Vec: VecTrait<NullableVec<FP16x16>, usize>,>() {
    
    // Test linalg solver
    let mut X_shape = ArrayTrait::<u32>::new();
    X_shape.append(3);
    X_shape.append(3);

    let mut X_data = ArrayTrait::<FP16x16>::new();
    X_data.append(FP16x16 { mag: 2, sign: false });
    X_data.append(FP16x16 { mag: 1, sign: false });
    X_data.append(FP16x16 { mag: 1, sign: true });
    X_data.append(FP16x16 { mag: 3, sign: true });
    X_data.append(FP16x16 { mag: 1, sign: true });
    X_data.append(FP16x16 { mag: 2, sign: false });
    X_data.append(FP16x16 { mag: 2, sign: true });
    X_data.append(FP16x16 { mag: 1, sign: false });
    X_data.append(FP16x16 { mag: 2, sign: false });

    let mut X = TensorTrait::<FP16x16>::new(shape: X_shape.span(), data: X_data.span());

    let mut y_shape = ArrayTrait::<u32>::new();
    y_shape.append(3);

    let mut y_data = ArrayTrait::<FP16x16>::new();
    y_data.append(FP16x16 { mag: 8, sign: false });
    y_data.append(FP16x16 { mag: 11, sign: true });
    y_data.append(FP16x16 { mag: 3, sign: true });

    let mut y = TensorTrait::<FP16x16>::new(shape: y_shape.span(), data: y_data.span());

    let mut sol = linalg_solve(X, y);
    test_tensor(sol);
}