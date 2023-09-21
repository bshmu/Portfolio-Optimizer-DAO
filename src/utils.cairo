mod optimizer_utils {
    use debug::PrintTrait;
    use option::OptionTrait;
    use array::{ArrayTrait, SpanTrait};
    use traits::{Into, TryInto, Index};
    use dict::Felt252DictTrait;
    use nullable::{NullableTrait, nullable_from_box, match_nullable, FromNullableResult};
    use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, FP16x16TensorMul, FP16x16TensorSub, FP16x16TensorDiv};
    use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16, FP16x16Add, FP16x16Div, FP16x16Mul, FP16x16Sub, FP16x16Impl};
    use orion::operators::tensor::core::ravel_index;
    use alexandria_data_structures::vec::{Felt252Vec, NullableVecImpl, NullableVec, VecTrait};

    struct MutTensor<FP16x16> {
        shape: Span<usize>,
        data: NullableVec<FP16x16>,
    } 

    trait MutTensorTrait<FP16x16> {
        fn new(shape: Span<usize>, data: NullableVec<FP16x16>) -> MutTensor<FP16x16>;
        fn at(ref self: @MutTensor<FP16x16>, indices: Span<usize>) -> FP16x16;
        fn set(ref self: @MutTensor<FP16x16>, indices: Span<usize>, value: FP16x16);
        fn to_tensor(ref self: @MutTensor<FP16x16>, indices: Span<usize>) -> Tensor<FP16x16>;
    }

    impl NullableVecCopy of Copy<NullableVec<FP16x16>>;
    impl NullableDictCopy of Copy<Felt252Dict<Nullable<FP16x16>>>;
    impl MutTensorImpl<> of MutTensorTrait<FP16x16> {
        fn new(shape: Span<usize>, data: NullableVec<FP16x16>) -> MutTensor<FP16x16> {
            MutTensor { shape, data }
        }

        fn at(ref self: @MutTensor<FP16x16>, indices: Span<usize>) -> FP16x16 {
            assert(indices.len() == (*self.shape).len(), 'Indices do not match dimensions');
            let mut data = *self.data;
            NullableVecImpl::get(ref data, ravel_index(*self.shape, indices)).unwrap()
        }

        fn set(ref self: @MutTensor<FP16x16>, indices: Span<usize>, value: FP16x16) {
            assert(indices.len() == (*self.shape).len(), 'Indices do not match dimensions');
            let mut data = *self.data;
            NullableVecImpl::set(ref data, ravel_index(*self.shape, indices), value)
        }

        fn to_tensor(ref self: @MutTensor<FP16x16>, indices: Span<usize>) -> Tensor<FP16x16> {
            assert(indices.len() == (*self.shape).len(), 'Indices do not match dimensions');
            let mut tensor_data = ArrayTrait::<FP16x16>::new();
            let mut i: u32 = 0;
            let n = self.data.len();
            let mut data: NullableVec<FP16x16> = *self.data;
            loop {
                if i == n {
                    break ();
                }
                let mut x_i = data.at(i);
                tensor_data.append(x_i);
                i += 1;
            };
            return TensorTrait::<FP16x16>::new(*self.shape, tensor_data.span());
        }
    }

    #[derive(Copy, Drop)]
    struct Matrix<
        impl FixedDict: Felt252DictTrait<FP16x16>,
        impl Vec: VecTrait<NullableVec<FP16x16>, usize>,
        impl VecDrop: Drop<NullableVec<FP16x16>>,
        impl VecCopy: Copy<NullableVec<FP16x16>>,
    > {
        rows: usize,
        cols: usize,
        data: NullableVec<FP16x16>,
    }

    fn forward_elimination<
        impl FixedDict: Felt252DictTrait<FP16x16>,
        impl Vec: VecTrait<NullableVec<FP16x16>, usize>,
        impl VecDrop: Drop<NullableVec<FP16x16>>,
        impl VecCopy: Copy<NullableVec<FP16x16>>,
    >(ref matrix: Matrix, ref vector: NullableVec<FP16x16>, n: usize) {
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

                let lhs = matrix.data.at(i * matrix.cols + row);
                let rhs = matrix.data.at(max_row * matrix.cols + row);
                if lhs > rhs {
                    max_row = i
                };

                i += 1;
            };

            matrix.data.set(row, matrix.data.at(max_row));
            matrix.data.set(max_row, matrix.data.at(row));
            vector.set(row, vector.at(max_row));
            vector.set(max_row, vector.at(row));

            // Check for singularity
            if matrix.data.at(row * matrix.cols + row) == 0 {
                panic(array!['Matrix is singular.'])
            }

            let mut i = row + 1;
            loop {
                if i == n {
                    break;
                };
                let factor = matrix.data.at(i * matrix.cols + row) / matrix.data.at(row * matrix.cols + row);
                let mut j = row;
                loop {
                    if j == n {
                        break;
                    }
                    matrix.data.set(matrix.data.at(i * matrix.cols + j), matrix.data.at(i * matrix.cols + j) - (factor * matrix.data.at(row * matrix.cols + j)));
                    j += 1;
                };
                vector.set(vector.at(i), vector.at(i) - (factor * vector.at(row)));
                i += 1;
            };
            row += 1;
        }

        // TODO: Map back the vector into a tensor
    }

    fn back_substitution<
        impl FixedDict: Felt252DictTrait<FP16x16>,
        impl Vec: VecTrait<NullableVec<FP16x16>, usize>,
        impl VecDrop: Drop<NullableVec<FP16x16>>,
        impl VecCopy: Copy<NullableVec<FP16x16>>,
    >(ref matrix: Matrix, ref vector: NullableVec<FP16x16>, n: usize) -> Tensor<FP16x16> {

        // Initialize the vector for the tensor data
        let mut x_items: Felt252Dict<Nullable<FP16x16>> = Default::default();
        let mut x_data: NullableVec<FP16x16> = NullableVec {items: x_items, len: n};

        // Loop through the array and assign the values
        let mut i: usize = n - 1;
        loop {
            x_data.set(x_data.at(i), vector.at(i));
            let mut j = i + 1;
            loop {
                if j == n {
                    break ();
                }
                x_data.set(x_data.at(i), matrix.data.at(i * matrix.cols + j) * x_data.at(j));
                j += 1;
            };
            x_data.set(x_data.at(i), x_data.at(i) / matrix.data.at(i * matrix.cols + i));
            if i == 0 {
                break ();
            }
            i -= 1;   
        };

        // Map back the vector into a tensor
        let mut x_mut: @MutTensor<FP16x16> = @MutTensor {shape: array![x_data.len()].span(), data: x_data};
        let x = x_mut.to_tensor(indices: *x_mut.shape);
        return x;
    }

    fn linalg_solve<
        impl FixedDict: Felt252DictTrait<FP16x16>,
        impl Vec: VecTrait<Felt252Vec<FP16x16>, usize>,
        impl VecDrop: Drop<Felt252Vec<FP16x16>>,
        impl VecCopy: Copy<Felt252Vec<FP16x16>>,
    >(X: Tensor<FP16x16>, y: Tensor<FP16x16>) -> Tensor<FP16x16> {
        // TODO: Map X and y to matrix and vector objects
        let n = *y.shape.at(0);
        forward_elimination(X, y, n);
        return back_substitution(X, y, n);
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
                break();
            }
            if X.shape.len() == 1 {
                let mut val = X.at(indices: array![i].span());
                val.mag.print();
            }
            else if X.shape.len() == 2 {
                let mut j = 0;
                loop {
                    if j == *X.shape.at(1) {
                        break ();
                    }
                    let mut val = X.at(indices: array![i, j].span());
                    val.mag.print();
                    j += 1;
                };
            }
            else {
                'Too many dims!'.print();
                break ();
            }
            i += 1;
        };
    }

}