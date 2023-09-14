mod optimizer_utils {
    use debug::PrintTrait;
    use option::OptionTrait;
    use array::{ArrayTrait, SpanTrait};
    use traits::{Into, TryInto, Index};
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
        implementations::fp16x16::core::{FP16x16Add, FP16x16Div, FP16x16DivEq, FP16x16Mul, FP16x16Sub, FP16x16SubEq, FP16x16Impl},
    };
    use alexandria_data_structures::vec::{VecTrait, Felt252VecImpl, Felt252Vec};

    fn exponential_weights(lambda_unscaled: u32, l: u32) -> Tensor<FixedType> {
        // Param lambda_unscaled (u32): factor for exponential weight calculation
        // Param l (felt252): length of vector to hold weights
        // Return (Tensor<FixedType>): 1D tensor of exponentially decaying fixed-point weights of length l
        let mut lambda = FixedTrait::new_unscaled(lambda_unscaled, false) / FixedTrait::new_unscaled(100, false); 
        let mut weights_array = ArrayTrait::<FixedType>::new(); // vector to hold weights
        let mut i: u32 = 0;
        loop {
            if i == l {
                break ();
            }
            let mut i_fp = FixedTrait::new_unscaled(i, false);
            let mut x = (FixedTrait::new_unscaled(1, false) - lambda) * (lambda.pow(i_fp));
            weights_array.append(x);
            i += 1;
        };

        // Convert the weights array into a tensor
        // Can shape be u32 and data be FixedType?
        let mut weights_len = ArrayTrait::<u32>::new();
        weights_len.append(l);
        let extra = ExtraParams {fixed_point: Option::Some(FixedImpl::FP16x16(()))};
        let mut weights_tensor = TensorTrait::<FixedType>::new(weights_len.span(), weights_array.span(), Option::Some(extra));
        return weights_tensor;
    }

    fn weighted_covariance(X: Tensor<FixedType>, weights: Tensor<FixedType>) -> Tensor<FixedType> {
        // Param X (Tensor<FixedType>): 2D Tensor of data to calculate covariance, shape (m,n)
        // Param weights (Tensor<FixedType>): Weights for covariance matrix
        // Return (Tensor<FixedType>): 2D Tensor covariance matrix, shape (n,n)
        
        // Get shape of array
        let m = *X.shape.at(0); // num rows
        let n = *X.shape.at(1); // num columns
        let l = *weights.shape.at(0); // length of weight vector
        assert(m == l, 'Data/weight length mismatch');

        // Transform weights vector into (l,l) diagonal matrix
        let mut W = diagonalize(weights);

        // Take dot product of W and X and center it
        // X_weighted = np.dot(W, X), shape = (m,n)
        let mut X_weighted = W.matmul(@X);

        // mean_weighted = (np.dot(weights, X) / np.sum(weights)), shape = (n,1)
        let extra = ExtraParams {fixed_point: Option::Some(FixedImpl::FP16x16(()))};
        let mut weights_T = weights.reshape(target_shape: array![1, *weights.shape.at(0)].span());
        let mut weights_dot_X = weights_T.matmul(@X);
        let mut weights_sum = *weights.reduce_sum(0, false).data.at(0);
        let mut mean_weighted_shape = ArrayTrait::<u32>::new();
        mean_weighted_shape.append(*weights_dot_X.shape.at(1));
        let mut mean_weighted_data = ArrayTrait::<FixedType>::new();
        let mut i: u32 = 0;
        loop {
            if i == *weights_dot_X.shape.at(1) {
                break ();
            }
            mean_weighted_data.append(*weights_dot_X.data.at(i) / weights_sum);
            i += 1;
        };

        let mean_weighted = TensorTrait::<FixedType>::new(mean_weighted_shape.span(), mean_weighted_data.span(), Option::Some(extra));

        // X_centered = X_weighted - mean_weighted, shape = (n,n)
        let mut X_centered_shape = ArrayTrait::<u32>::new();
        X_centered_shape.append(n);
        X_centered_shape.append(n);
        let mut X_centered_data = ArrayTrait::<FixedType>::new();
        let mut row: u32 = 0;
        loop {
            if row == n {
                break ();
            }
            let mut row_i: u32 = 0;
            loop {
                if row_i == n {
                    break ();
                }
                X_centered_data.append(*X_weighted.data.at(row_i) - *mean_weighted.data.at(row));
                row_i += 1;
            };
            row += 1;
        };
        let X_centered = TensorTrait::<FixedType>::new(X_centered_shape.span(), X_centered_data.span(), Option::Some(extra));

        // Calculate covariance matrix
        // covariance_matrix = centered_data.T.dot(centered_data) / (np.sum(weights) - 1)
        let mut X_centered_T = X_centered.transpose(axes: array![1, 0].span());
        let mut Cov_X_num =  X_centered_T.matmul(@X_centered);
        let mut Cov_X_den = *weights.reduce_sum(0, false).data.at(0) - FixedTrait::new_unscaled(1, false);
        let mut Cov_X_shape = Cov_X_num.shape;
        let mut Cov_X_data = ArrayTrait::<FixedType>::new();
        i = 0;
        loop {
            if (i == *Cov_X_shape.at(0) * Cov_X_shape.len()) {
                break ();
            }
            Cov_X_data.append(*Cov_X_num.data.at(i) / Cov_X_den);
            i += 1;
        };
        
        let Cov_X = TensorTrait::<FixedType>::new(Cov_X_shape, Cov_X_data.span(), Option::Some(extra));
        return Cov_X;
    }

    fn rolling_covariance(df: Tensor<FixedType>, lambda: u32, w: u32) -> Array<Tensor<FixedType>> {
        // Param df (Tensor<FixedType>): time series of historical data, shape (m,n)
        // Param lambda (u32): factor for exponential weight calculation
        // Param w (felt252): length of rolling window
        // Return Array<Tensor<FixedType>> -- array of rolling covariance matrices

        // Get shape of data
        let m = *df.shape.at(0); // num rows
        let n = *df.shape.at(1); // num columns
        let weights = exponential_weights(lambda, w);
        let extra = ExtraParams {fixed_point: Option::Some(FixedImpl::FP16x16(()))};

        // Loop through the data and calculate the covariance on each subset
        let mut results = ArrayTrait::<Tensor<FixedType>>::new();
        let mut row = 0;
        loop {
            row.print();
            if row == (m - w + 1) {
                break ();
            }

            // Get subset of data
            let mut subset_shape = ArrayTrait::<u32>::new();
            subset_shape.append(w);
            subset_shape.append(n);
            let mut subset_data = ArrayTrait::<FixedType>::new();
            let mut i = row * n;
            loop {
                if i == (row + w) * n {
                    break ();
                }
                subset_data.append(*df.data.at(i));
                i += 1;
            };
            let mut subset = TensorTrait::<FixedType>::new(subset_shape.span(), subset_data.span(), Option::Some(extra));

            // Calculate covariance matrix on the subset and append
            let mut Cov_i = weighted_covariance(subset, weights);
            results.append(Cov_i);
            row += 1;
        };

        return results;
    }

    fn diagonalize(X_input: Tensor::<FixedType>) -> Tensor::<FixedType> {
        // Make sure input tensor is 1D
        assert(X_input.shape.len() == 1, 'Input tensor is not 1D.');

        // 2D Shape for output tensor
        let mut X_output_shape = ArrayTrait::<u32>::new();
        let n = *X_input.shape.at(0);
        X_output_shape.append(n);
        X_output_shape.append(n);        
        
        // Data
        let mut X_output_data = ArrayTrait::<FixedType>::new();
        let mut i: u32 = 0;
        loop {
            if i == n {
                break ();
            }
            let mut j = 0;
            loop {
                if j == n {
                    break ();
                }
                if i == j {
                    X_output_data.append(*X_input.data.at(i));
                }
                else {
                    X_output_data.append(FixedTrait::new_unscaled(0, false));
                }
                j += 1;
            };
            i += 1;
        };

        // Return final diagonal matrix
        let extra = ExtraParams {fixed_point: Option::Some(FixedImpl::FP16x16(()))};
        return TensorTrait::<FixedType>::new(X_output_shape.span(), X_output_data.span(), Option::Some(extra));
    }

    #[derive(Copy, Drop)]
    struct Matrix<
        impl FixedDict: Felt252DictTrait<FixedType>,
        impl Vec: VecTrait<Felt252Vec<FixedType>, usize>,
        impl VecDrop: Drop<Felt252Vec<FixedType>>,
        impl VecCopy: Copy<Felt252Vec<FixedType>>,
    > {
        rows: usize,
        cols: usize,
        data: Felt252Vec<FixedType>,
    }

    fn forward_elimination<
        impl FixedDict: Felt252DictTrait<FixedType>,
        impl Vec: VecTrait<Felt252Vec<FixedType>, usize>,
        impl VecDrop: Drop<Felt252Vec<FixedType>>,
        impl VecCopy: Copy<Felt252Vec<FixedType>>,
    >(ref matrix: Matrix, ref vector: Felt252Vec<FixedType>, n: usize) {
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
        impl FixedDict: Felt252DictTrait<FixedType>,
        impl Vec: VecTrait<Felt252Vec<FixedType>, usize>,
        impl VecDrop: Drop<Felt252Vec<FixedType>>,
        impl VecCopy: Copy<Felt252Vec<FixedType>>,
    >(ref matrix: Matrix, ref vector: Felt252Vec<FixedType>, n: usize) -> Tensor<FixedType> {
        // Initialize array of zeros
        let mut x_items: Felt252Dict<FixedType> = Default::default();
        let mut i = 0;
        loop {
            if i == n {
                break ();
            }
            x_items.insert(i.into(), FixedTrait::new_unscaled(0, false));
            i += 1;
        };
        let mut x: Felt252Vec<FixedType> = Felt252Vec {items: x_items, len: n};

        // Loop through the array and assign the values
        i = n - 1;
        loop {
            x.set(x.at(i), vector.at(i));
            let mut j = i + 1;
            loop {
                if j == n {
                    break ();
                }
                x.set(x.at(i - 1), matrix.data.at(i * matrix.cols + j) * x.at(j));
                j += 1;
            };
            x.set(x.at(i), x.at(i) / matrix.data.at(i * matrix.cols + i));
            if i == 0 {
                break ();
            }
            i -= 1;   
        };

        // TODO: Map back the vector into a tensor
    }

    fn linalg_solve<
        impl FixedDict: Felt252DictTrait<FixedType>,
        impl Vec: VecTrait<Felt252Vec<FixedType>, usize>,
        impl VecDrop: Drop<Felt252Vec<FixedType>>,
        impl VecCopy: Copy<Felt252Vec<FixedType>>,
    >(X: Tensor<FixedType>, y: Tensor<FixedType>) -> Tensor<FixedType> {
        // TODO: Map X and y to matrix and vector objects
        let n = *y.shape.at(0);
        forward_elimination(X, y, n);
        return back_substitution(X, y, n);
    }

    fn test_tensor(X: Tensor::<FixedType>) {
        // 'Test...'.print();
        // 'Len...'.print();
        // X.data.len().print();
        // 'Vals...'.print();
        // Print x by rows
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