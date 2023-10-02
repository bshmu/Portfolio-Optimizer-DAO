
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
use alexandria_data_structures::vec::{Felt252Vec, VecTrait};
use utils::optimizer_utils::{forward_elimination,
                             back_substitution,
                             linalg_solve,
                             test_tensor};

// Nullable Vec Impl
struct NullableVec<FP16x16> {
    items: Felt252Dict<Nullable<FP16x16>>,
    len: usize,
}

impl DestructNullableVec<FP16x16, impl FP16x16Drop: Drop<FP16x16>> of Destruct<NullableVec<FP16x16>> {
    fn destruct(self: NullableVec<FP16x16>) nopanic {
        self.items.squash();
    }
}

impl NullableVecImpl<FP16x16, 
                     impl FP16x16Drop: Drop<FP16x16>, 
                     impl FP16x16Copy: Copy<FP16x16>> 
                     of VecTrait<NullableVec<FP16x16>, FP16x16> {
    fn new() -> NullableVec<FP16x16> {
        NullableVec { items: Default::default(), len: 0 }
    }

    fn get(ref self: NullableVec<FP16x16>, index: usize) -> Option<FP16x16> {
        if index < self.len() {
            Option::Some(self.items.get(index.into()).deref())
        } else {
            Option::None
        }
    }

    fn at(ref self: NullableVec<FP16x16>, index: usize) -> FP16x16 {
        assert(index < self.len(), 'Index out of bounds');
        self.items.get(index.into()).deref()
    }

    fn push(ref self: NullableVec<FP16x16>, value: FP16x16) -> () {
        self.items.insert(self.len.into(), nullable_from_box(BoxTrait::new(value)));
        self.len = integer::u32_wrapping_add(self.len, 1_usize);
    }

    fn set(ref self: NullableVec<FP16x16>, index: usize, value: FP16x16) {
        assert(index < self.len(), 'Index out of bounds');
        self.items.insert(index.into(), nullable_from_box(BoxTrait::new(value)));
    }

    fn len(self: @NullableVec<FP16x16>) -> usize {
        *self.len
    }
}

#[test]
#[available_gas(99999999999999999)]
fn test(){
    
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