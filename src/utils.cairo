mod optimizer_utils {
    use debug::PrintTrait;
    use option::OptionTrait;
    use array::{ArrayTrait, SpanTrait};
    use traits::{Into, TryInto, Index};
    use nullable::{NullableTrait, nullable_from_box, match_nullable, FromNullableResult};
    use orion::operators::tensor::{
        Tensor, TensorTrait, FP16x16Tensor, FP16x16TensorMul, FP16x16TensorSub, FP16x16TensorDiv
    };
    use orion::numbers::fixed_point::implementations::fp16x16::core::{
        FP16x16, FP16x16Add, FP16x16Div, FP16x16Mul, FP16x16Sub, FP16x16Impl
    };
    use orion::operators::tensor::core::ravel_index;
    use alexandria_data_structures::vec::{Felt252Vec, NullableVecImpl, NullableVec, VecTrait};

    impl Felt252DictNullableDrop of Drop<Felt252Dict<Nullable<FP16x16>>>;
    impl MutTensorDrop of Drop<MutTensor>;
    impl NullableVecDrop of Drop<NullableVec<FP16x16>>;
    impl NullableVecCopy of Copy<NullableVec<FP16x16>>;
    impl NullableDictCopy of Copy<Felt252Dict<Nullable<FP16x16>>>;

    struct MutTensor {
        shape: Span<usize>,
        data: NullableVec<FP16x16>,
    }

    #[generate_trait]
    impl FP16x16MutTensorImpl of MutTensorTrait {
        fn new(shape: Span<usize>, data: NullableVec<FP16x16>) -> MutTensor {
            MutTensor { shape, data }
        }

        fn at(ref self: @MutTensor, indices: Span<usize>) -> FP16x16 {
            assert(indices.len() == (*self.shape).len(), 'Indices do not match dimensions');
            let mut data = *self.data;
            NullableVecImpl::get(ref data, ravel_index(*self.shape, indices)).unwrap()
        }

        fn set(ref self: @MutTensor, indices: Span<usize>, value: FP16x16) {
            assert(indices.len() == (*self.shape).len(), 'Indices do not match dimensions');
            let mut data = *self.data;
            NullableVecImpl::set(ref data, ravel_index(*self.shape, indices), value)
        }

        fn get_rows(ref self: MutTensor) -> usize {
            let l = self.shape;
            l.len()
        }

        fn get_cols(ref self: MutTensor) -> usize {
            let l = self.shape;
            *l.at(0)
        }

        fn to_tensor(ref self: MutTensor, indices: Span<usize>) -> Tensor<FP16x16> {
            assert(indices.len() == (self.shape).len(), 'Indices do not match dimensions');
            let mut tensor_data = ArrayTrait::<FP16x16>::new();
            let mut i: u32 = 0;
            let n = self.data.len();
            let mut data: NullableVec<FP16x16> = self.data;
            loop {
                if i == n {
                    break ();
                }
                let mut x_i = data.at(i);
                tensor_data.append(x_i);
                i += 1;
            };
            return TensorTrait::<FP16x16>::new(self.shape, tensor_data.span());
        }
    }

    fn forward_elimination<impl Vec: VecTrait<NullableVec<FP16x16>, usize>,>(ref X: MutTensor, ref y: MutTensor, n: usize) {
        let cols_X = X.get_cols();
        let mut row: usize = 0;
        loop {
            if row == n {
                break;
            };

            let mut max_row = row;
            let mut i = row + 1;
            loop {
                if i == n {
                    break;
                };

                let mut lhs: FP16x16 = X.data.at(i * cols_X + row);
                let mut rhs: FP16x16 = X.data.at(max_row * cols_X + row);
                if lhs > rhs {
                    max_row = i
                };

                i += 1;
            };

            let mut X_new_val: FP16x16 = X.data.at(max_row);
            let mut X_old_val: FP16x16 = X.data.at(row);
            let mut y_new_val: FP16x16 = y.data.at(max_row);
            let mut y_old_val: FP16x16 = y.data.at(row);

            X.data.set(row, X_new_val);
            X.data.set(max_row, X_old_val);
            y.data.set(row, y_new_val);
            y.data.set(max_row, y_old_val);

            // Check for singularity
            let mut X_check_val: usize = row * cols_X + row;
            if X_check_val == 0 {
                panic(array!['Matrix is singular.'])
            }

            let mut i = row + 1;
            loop {
                if i == n {
                    break;
                };
                let factor = X.data.at(i * cols_X + row) / X.data.at(row * cols_X + row);
                let mut j = row;
                loop {
                    if j == n {
                        break;
                    }
                    X.data.set(X.data.at(i * cols_X + j),
                               X.data.at(i * cols_X + j) - (factor * X.data.at(row * cols_X + j)));
                    j += 1;
                };
                let mut y_set_val: FP16x16 = y.data.at(row);
                y.data.set(y.data.at(i), y.data.at(i) - (factor * y_set_val));
                i += 1;
            };
            row += 1;
        }
    }

    fn back_substitution<impl Vec: VecTrait<NullableVec<FP16x16>, usize>,>(ref X: MutTensor, ref y: MutTensor, n: usize) -> Tensor<FP16x16> {

        // Initialize the vector for the tensor data
        let mut x_items: Felt252Dict<Nullable<FP16x16>> = Default::default();
        let mut x_data: NullableVec<FP16x16> = NullableVec { items: x_items, len: n };

        // Loop through the array and assign the values
        let cols_X = X.get_cols();
        let mut i: usize = n - 1;
        loop {
            x_data.set(x_data.at(i), i);
            let mut j = i + 1;
            loop {
                if j == n {
                    break ();
                }
                let mut x_data_val_0: FP16x16 = X.data.at(i * cols_X + j) * x_data.at(j);
                x_data.set(x_data.at(i), x_data_val_0);
                j += 1;
            };
            let mut x_data_val_1: FP16x16 = x_data.at(i) / X.data.at(i * cols_X + i);
            x_data.set(x_data.at(i), x_data_val_1);
            if i == 0 {
                break ();
            }
            i -= 1;
        };

        // Map back the vector into a tensor
        X.to_tensor(indices: X.shape)
    }

    fn linalg_solve<impl Vec: VecTrait<NullableVec<FP16x16>, usize>,>(X: Tensor<FP16x16>, y: Tensor<FP16x16>) -> Tensor<FP16x16> {
        // Assert X and y are the same length
        let n = *X.shape.at(0);
        assert(n == *y.shape.at(0), 'Matrix/vector dim mismatch');

        // Map X and y to MutTensor objects
        let mut X_items: Felt252Dict<Nullable<FP16x16>> = Default::default();
        let mut X_data: NullableVec<FP16x16> = NullableVec { items: X_items, len: n };
        let mut y_items: Felt252Dict<Nullable<FP16x16>> = Default::default();
        let mut y_data: NullableVec<FP16x16> = NullableVec { items: y_items, len: n };
        let mut i: usize = 0;
        loop {
            if i == n {
                break ();
            }
            let mut j: usize = 0;
            loop {
                if j == n {
                    break ();
                }
                X_data.set(j, *X.data.at(j));
                j += 1;
            };
            y_data.set(i, *y.data.at(i));
            i += 1;
        };
        let mut X_mut = MutTensor { shape: X.shape, data: X_data };
        let mut y_mut = MutTensor { shape: y.shape, data: y_data };

        let n = *y.shape.at(0);
        forward_elimination(ref X_mut, ref y_mut, n);
        return back_substitution(ref X_mut, ref y_mut, n);
    }

    fn test_tensor(X: Tensor::<FP16x16>) {
        'Testing tensor...'.print();
        'Len...'.print();
        X.data.len().print();

        // Print x by rows
        'Vals...'.print();
        let mut i = 0;
        loop {
            if i == *X.shape.at(0) {
                break ();
            }
            if X.shape.len() == 1 {
                let mut val = X.at(indices: array![i].span());
                val.mag.print();
            } else if X.shape.len() == 2 {
                let mut j = 0;
                loop {
                    if j == *X.shape.at(1) {
                        break ();
                    }
                    let mut val = X.at(indices: array![i, j].span());
                    val.mag.print();
                    j += 1;
                };
            } else {
                'Too many dims!'.print();
                break ();
            }
            i += 1;
        };
    }
}