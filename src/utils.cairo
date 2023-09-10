mod optimizer_utils {
    use debug::PrintTrait;
    use option::OptionTrait;
    use array::{ArrayTrait, SpanTrait};
    use traits::{Into, TryInto};
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

    fn forward_elimination(X: Tensor::<FixedType>, y: Tensor::<FixedType>) -> (Tensor::<FixedType>, Tensor::<FixedType>) {
        // Param X (Tensor::<FixedType>): 2D tensor
        // Param y (Tensor::<FixedType>): 1D tensor
        // Returns (Tensor::<FixedType>, Tensor::<FixedType>) -- X in upper triangular form, y adjusted

        // Forward elimination --> Transform X to upper triangular form
        let n = *X.shape.at(0);
        let l = *y.shape.at(0);
        assert(n == l, 'num_rows(X) != len(y)');
        let mut row = 0;
        let extra = ExtraParams {fixed_point: Option::Some(FixedImpl::FP16x16(()))};
        
        // Initialize XUT
        let mut XUT_initial_data = ArrayTrait::<FixedType>::new();
        let mut i = 0;
        loop {
            if i == X.data.len() {
                break ();
            }
            XUT_initial_data.append(FixedTrait::new_unscaled(0, false));
            i += 1;
        };
        let mut XUT = TensorTrait::<FixedType>::new(X.shape, XUT_initial_data.span(), Option::Some(extra));


        // Initialize y_out
        let mut y_out_initial = ArrayTrait::<FixedType>::new();
        i = 0;
        loop {
            if i == l {
                break ();
            }
            y_out_initial.append(FixedTrait::new_unscaled(0, false));
            i += 1;
        };
        let mut y_out = TensorTrait::<FixedType>::new(y.shape, y_out_initial.span(), Option::Some(extra));

        let mut Xy = (XUT, y_out);

        // Construct XUT
        Xy = loop {
            if (row == n) {
                break Xy;
            }

            // 'Row:'.print();
            // row.print();

            // First loop rearranges the matrix
            // For each column, find the row with largest absolute value
            let mut max_row = row;
            let mut i = row + 1;
            max_row = loop {
                if (i == n) {
                    break max_row;
                }
                if (X.at(indices: array![i, row].span()).mag > X.at(indices: array![max_row, row].span()).mag) {
                    max_row = i;
                }
                i += 1;
            };

            // 'Max row:'.print();
            // max_row.print();

            // Move the max row to the top of the matrix
            let mut XUT_intermediate_data = ArrayTrait::<FixedType>::new();
            i = 0;
            loop {
                if i == n {
                    break ();
                }
                let mut j = 0; 
                loop {
                    if j == n {
                        break ();
                    }
                    if i == max_row {
                        XUT_intermediate_data.append(X.at(indices: array![row, j].span()));
                    }
                    else if i == row {
                        XUT_intermediate_data.append(X.at(indices: array![max_row, j].span()));
                    }
                    else {
                        XUT_intermediate_data.append(X.at(indices: array![i, j].span()));
                    }
                    j += 1;
                };
                i += 1;
            };
            let mut XUT_intermediate = TensorTrait::<FixedType>::new(X.shape, XUT_intermediate_data.span(), Option::Some(extra));

            // Rearrange the y vector
            let mut y_out_intermediate_data = ArrayTrait::<FixedType>::new();
            i = 0;
            loop {
                if i == n {
                    break ();
                }   
                if i == max_row {
                    y_out_intermediate_data.append(y.at(indices: array![row].span()));
                }
                else if i == row {
                    y_out_intermediate_data.append(y.at(indices: array![max_row].span()));
                }
                else {
                    y_out_intermediate_data.append(y.at(indices: array![i].span()));
                }
                i += 1;
            };
            let mut y_out_intermediate = TensorTrait::<FixedType>::new(y.shape, y_out_intermediate_data.span(), Option::Some(extra));

            // Check for singularity
            assert(XUT_intermediate.at(indices: array![row, row].span()).mag != 0, 'matrix is singular');

            // test_tensor(XUT_intermediate);
            // test_tensor(y_out_intermediate);

            // Remove zeros below diagonal
            let mut XUT_final_data = ArrayTrait::<FixedType>::new();
            let mut y_out_final_data = ArrayTrait::<FixedType>::new();
            i = 0;
            loop {
                if i == n {
                    // 'len(XUT):'.print();
                    // XUT_final_data.len().print();
                    break ();
                }
                else if i >= row + 1 {
                    let mut factor = XUT_intermediate.at(indices: array![i, row].span()) / XUT_intermediate.at(indices: array![row, row].span());
                    let mut j = 0;
                    loop {
                        if j == n {
                            break ();
                        }
                        if j >= row {
                            let mut val = XUT_intermediate.at(indices: array![i, j].span()) - (factor * XUT_intermediate.at(indices: array![row, j].span()));
                            XUT_final_data.append(val);
                        }
                        else {
                            XUT_final_data.append(XUT_intermediate.at(indices: array![i, j].span()));
                        }
                        j += 1;
                    };
                    let mut val_y = y_out_intermediate.at(indices: array![i].span()) - (factor * y_out_intermediate.at(indices: array![row].span()));
                    y_out_final_data.append(val_y);
                }
                else {
                    let mut j = 0;
                    loop {
                        if j == n {
                            break ();
                        }
                        XUT_final_data.append(XUT_intermediate.at(indices: array![i, j].span()));
                        j += 1;
                    };
                    y_out_final_data.append(y_out_intermediate.at(indices: array![i].span()));
                }
                i += 1;
            };
            XUT = TensorTrait::<FixedType>::new(X.shape, XUT_final_data.span(), Option::Some(extra));
            y_out = TensorTrait::<FixedType>::new(y.shape, y_out_final_data.span(), Option::Some(extra));

            Xy = (XUT, y_out);
            
            row += 1;
        };

        return Xy;
    }

    fn back_substitution(X: Tensor::<FixedType>, y: Tensor::<FixedType>) -> Tensor::<FixedType> {
        // Param X (Tensor::<FixedType>): 2D tensor
        // Param y (Tensor::<FixedType>): 1D tensor
        // Return sln (Tensor::<FixedType>): Uses back substitution to solve the system for the upper triangular matrix

        // Initialize a tensor of zeros that will store the solutions to the system
        let n = *X.shape.at(0);
        let l = *y.shape.at(0);
        assert(n == l, 'num_rows(X) != len(y)');
        let extra = ExtraParams {fixed_point: Option::Some(FixedImpl::FP16x16(()))};

        // Initialize sln vector
        let mut sln_data_initial = ArrayTrait::<FixedType>::new();
        let mut c = 0;
        loop {
            if c == n {
                break ();
            }
            sln_data_initial.append(FixedTrait::new_unscaled(0, false));
            c += 1;
        };
        let mut sln = TensorTrait::<FixedType>::new(array![n].span(), sln_data_initial.span(), Option::Some(extra));

        // Backward iteration
        let mut sln_data = ArrayTrait::<FixedType>::new();
        c = n + 1;
        loop {
            if c == 0 {
                break ();
            }
            let mut i = c - 1;

            // Begin by assuming the solution sln[i] for the current row is the corresponding value in the vector y
            let mut sln_i = y.at(indices: array![i].span());

            // Iterate over columns to the right of the diagonal for the current row
            let mut j = i + 1;
            loop {
                if j == n {
                    break ();
                }
                // Subtract the product of the known solution sln[j] and the matrix coefficient from the current solution sln[i]
                // How can we access previous solutions in the middle of the loop???
                let mut val = X.at(indices: array![i, j].span()) * *sln_data.at(j - (i + 1));
                sln_i -= val;
            };

            // Normalize to get the actual value
            sln_i /= X.at(indices: array![i, i].span());
            sln_data.append(sln_i);
            
            c -= 1;
        };

        // Return the solution
        return sln;
    }

    fn linalg_solve(X: Tensor::<FixedType>, y: Tensor::<FixedType>) -> Tensor::<FixedType> {
        // Solve the system of linear equations using Gaussian elimination
        let Xy = forward_elimination(X, y);
        let (XUT, y_out) = Xy; 
        return back_substitution(XUT, y_out);
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