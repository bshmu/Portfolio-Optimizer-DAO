mod optimizer_utils {
    use debug::PrintTrait;
    use option::OptionTrait;
    use array::{ArrayTrait, SpanTrait};
    use traits::{Into, TryInto};
    use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
    use orion::numbers::fixed_point::implementations::fp8x23::core::{FP8x23Add, FP8x23Div, FP8x23Mul, FP8x23Sub, FP8x23Impl};
    use orion::operators::tensor::core::{Tensor, TensorTrait, ExtraParams};
    use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};


    fn exponential_weights(l: u32, lambda_unscaled: u32) -> Tensor<FixedType> {
        // Param l (felt252): length of vector to hold weights
        // Param lambda_unscaled (u32): factor for exponential weight calculation
        // Return (Tensor<FixedType>): 1D tensor of exponentially decaying fixed-point weights of length l
        let mut lambda = FixedTrait::new_unscaled(lambda_unscaled, false) / FixedTrait::new_unscaled(100, false); 
        let mut weights_array = ArrayTrait::<FixedType>::new(); // vector to hold weights
        let mut i: u32 = 0;
        let one: u32 = 1;
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
        let mut weights_tensor = TensorTrait::<FixedType>::new(weights_len.span(), weights_array.span(), Option::<ExtraParams>::None(()));
        return weights_tensor;
    }

    // fn weighted_covariance(X: Tensor<FixedType>, weights: Tensor<FixedType>) -> Tensor<FixedType> {
    //     // Param X (Tensor<FixedType>): 2D Tensor of data to calculate covariance, shape (m,n)
    //     // Param weights (Tensor<FixedType>): Weights for covariance matrix
    //     // Return (Tensor<FixedType>): 2D Tensor covariance matrix, shape (n,n)

    //     // Get shape of array
    //     let m = *X.shape.at(0); // num rows
    //     let n = *X.shape.at(1); // num columns
    //     let l = *weights.shape.at(0); // length of weight vector
    //     assert(n == l, 'Data/weight length mismatch');

    //     // Transform weights vector into (n,n) diagonal matrix
    //     let mut W = diagonalize(weights);

    //     // Take dot product of W and X and center it
    //     let mut X_weighted = W.matmul(@X); // np.dot(W, X)
    //     let mut mean_weighted = weights.matmul(@X) / weights.reduce_sum(0, false); // np.dot(weights, X) / np.sum(weights)
    //     let mut X_centered = X_weighted - mean_weighted; //  X_weighted - mean_weighted
        
    //     // Calculate covariance matrix
    //     // covariance_matrix = centered_data.T.dot(centered_data) / (np.sum(weights) - 1)
    //     let mut X_centered_T = X_centered.transpose();
    //     let Cov_X = X_centered_T.matmul(@X_centered) / (weights.reduce_sum(0, false) - 1);
        
    //     return Cov_X;
    // }

    // fn rolling_covariance(df: Tensor<FixedType>, lambda: u32, w: u32) -> Array<Tensor<FixedType>> {
    //     // Get shape of data
    //     let mut m = *df.shape.at(0); // num rows
    //     let mut n = *df.shape.at(1); // num columns
    //     let mut weights = exponential_weights(w);
    //     let mut results = ArrayTrait::<Tensor<FixedType>>::new();

    //     // Loop through the data and calculate the covariance on each subset
    //     let mut i = 0;
    //     loop {
    //         if i == (n - w + 1) {
    //             break ();
    //         }

    //         // Get subset of data
    //         let mut subset_array = ArrayTrait::<FixedType>::new();
    //         let mut j = 0;
    //         loop {
    //             if j == (w - 1) {
    //                 break ();
    //             }
    //             subset_array.append(*df.data.at(j));
    //             j += 1;
    //         };

    //         // Calculate covariance matrix and append
    //         let mut Cov_i = weighted_covariance(subset, weights);
    //         results.append(Cov_i);
    //         i += 1;
    //     };

    //     return results;
    // }

    // fn diagonalize(X_input: Tensor::<FixedType>) -> Tensor::<FixedType> {
    //     // Shape
    //     let mut X_output_shape = ArrayTrait::<u32>::new();
    //     let n = *X_input.shape.at(1); 
    //     let mut i_shape: u32 = 0;
    //     loop {
    //         if i_shape == n {
    //             break ();
    //         }
    //         X_output_shape.append(n);
    //         i_shape += 1;
    //     };
        
    //     // Data
    //     let mut X_output_data = ArrayTrait::<FixedType>::new();
    //     let mut i = 0;
    //     let mut j = 0;
    //     loop {
    //         if i == n * n {
    //             break ();
    //         }
    //         if i == n * j {
    //             X_output_data.append(*X_input.data.at(i));
    //             j += 1;
    //         }
    //         else {
    //             X_output_data.append(FixedTrait::new(8388608, false));
    //         }
    //         i += 1;
    //     };

    //     // Return final diagonal matrix
    //     let mut X_extra = Option::<ExtraParams>::None(());
    //     return TensorTrait::<FixedType>::new(X_output_shape.span(), X_output_data.span(), X_extra);
    // }

    // fn forward_elimination(X: Tensor::<FixedType>, y: Tensor::<FixedType>) {
    //     // Forward elimination --> Transform X to upper triangular form
    //     let n = *X.shape.at(0);
    //     let mut row = 0;
    //     let mut matrix = ArrayTrait::new();
    //     loop {
    //         if (row == n) {
    //             break ();
    //         }

    //         // For each column, find the row with largest absolute value
    //         let mut max_row = row;
    //         let mut i = row + 1;
    //         loop {
    //             if (i == n) {
    //                 break ();
    //             }

    //             if (X.at(indices: !array[i, row].span()).abs() > X.at(indices: !array[max_row, row].span()).abs()) {
    //                 max_row = i;
    //             }

    //             i += 1;
    //         };

    //         // Swap the max row with the current row to make sure the largest value is on the diagonal
    //         // need to replicate X[row], X[max_row] = X[max_row], X[row]
    //         // How do this using Orion??
    //         X.at(indices: !array[row].span()) = X.at(indices: !array[max_row].span());
    //         X.at(indices: !array[max_row].span()) = X.at(indices: !array[row].span());
    //         y.at(indices: !array[row].span()) = y.at(indices: !array[max_row].span());
    //         y.at(indices: !array[max_row].span()) = y.at(indices: !array[row].span());
            

    //         // Check for singularity
    //         assert(X.at(indices: !array[row, row].span()) != 0, 'matrix is singular');
            
    //         // Eliminate values below diagonal
    //         let mut j = row + 1;
    //         loop {
    //             if (j == n) {
    //                 break ();
    //             }

    //             let mut factor = X.at(indices: !array[j, row].span()) / X.at(indices: !array[row, row].span());
    //             let mut k = row;
    //             loop {
    //                 if (k == n) {
    //                     break ();
    //                 }
    //                 X.at(indices: !array[j, k].span()) -= factor * X.at(indices: !array[row, k].span());

    //                 k += 1;
    //             };
    //             y.at(indices: !array[j].span()) -= factor * y.at(indices: !array[row].span());

    //             j += 1;
    //         };

    //         row += 1;
    //     };
    // }

    // fn back_substitution(X: Tensor::<FixedType>, y: Tensor::<FixedType>) {
    //     // Uses back substitution to solve the system for the upper triangular matrix found in forward_elimination()

    //     // Initialize a tensor of zeros that will store the solutions to the system
    //     let n = *X.shape.at(0);
    //     let mut sln_data = ArrayTrait::new();
    //     let mut c = 0;
    //     loop {
    //         if c == n {
    //             break ();
    //         }
    //         sln_data.append(0);
    //         c += 1;
    //     };
    //     let mut sln = TensorTrait::<u32>::new(shape: !array[n].span(), data: sln_data.span(),Option::<ExtraParams>::None(()));

    //     // Backward iteration
    //     let mut row = n;
    //     loop {
    //         if row == 0 {
    //             break ();
    //         }

    //         // Begin by assuming the solution x[row] for the current row is the corresponding value in the vector y
    //         sln.at(indices: !array[row].span()) = y.at(indices: !array[row].span());

    //         // Iterate over columns to the right of the diagonal for the current row
    //         let mut i = row + 1;
    //         loop {
    //             if i == n {
    //                 break ();
    //             }
    //             // Subtract the product of the known solution x[i] and the matrix coefficient from the current solution x[row]
    //             sln.at(indices: !array[row].span()) -= X.at(indices: !array[row, i].span()) * sln.at(indices: !array[i].span());
    //         };

    //         // Normalize to get the actual value
    //         sln.at(indices: !array[row].span()) /= X.at(indices: !array[row, row].span())
    //         row -= 1;
    //     };

    //     // Return the solution
    //     return sln;
    // }

    // fn linalg_solve(X: Tensor::<FixedType>, y: Tensor::<FixedType>) -> Tensor::<FixedType> {
    //     // Solve the system of linear equations using Gaussian elimination
    //     let n = *X.shape.at(0);
    //     forward_elimination(X, y);
    //     return back_substitution(X, y);
    // }

}