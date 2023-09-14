// fn forward_elimination(X: Tensor::<FixedType>, y: Tensor::<FixedType>) -> (Tensor::<FixedType>, Tensor::<FixedType>) {
//     // Param X (Tensor::<FixedType>): 2D tensor
//     // Param y (Tensor::<FixedType>): 1D tensor
//     // Returns (Tensor::<FixedType>, Tensor::<FixedType>) -- X in upper triangular form, y adjusted

//     // Forward elimination --> Transform X to upper triangular form
//     let n = *X.shape.at(0);
//     let l = *y.shape.at(0);
//     assert(n == l, 'num_rows(X) != len(y)');
//     let mut row = 0;
//     let extra = ExtraParams {fixed_point: Option::Some(FixedImpl::FP16x16(()))};
    
//     // Initialize XUT
//     let mut XUT_initial_data = ArrayTrait::<FixedType>::new();
//     let mut i = 0;
//     loop {
//         if i == X.data.len() {
//             break ();
//         }
//         XUT_initial_data.append(FixedTrait::new_unscaled(0, false));
//         i += 1;
//     };
//     let mut XUT = TensorTrait::<FixedType>::new(X.shape, XUT_initial_data.span(), Option::Some(extra));


//     // Initialize y_out
//     let mut y_out_initial = ArrayTrait::<FixedType>::new();
//     i = 0;
//     loop {
//         if i == l {
//             break ();
//         }
//         y_out_initial.append(FixedTrait::new_unscaled(0, false));
//         i += 1;
//     };
//     let mut y_out = TensorTrait::<FixedType>::new(y.shape, y_out_initial.span(), Option::Some(extra));

//     let mut Xy = (XUT, y_out);

//     // Construct XUT
//     Xy = loop {
//         if (row == n) {
//             break Xy;
//         }

//         // 'Row:'.print();
//         // row.print();

//         // First loop rearranges the matrix
//         // For each column, find the row with largest absolute value
//         let mut max_row = row;
//         let mut i = row + 1;
//         max_row = loop {
//             if (i == n) {
//                 break max_row;
//             }
//             if (X.at(indices: array![i, row].span()).mag > X.at(indices: array![max_row, row].span()).mag) {
//                 max_row = i;
//             }
//             i += 1;
//         };

//         // 'Max row:'.print();
//         // max_row.print();

//         // Move the max row to the top of the matrix
//         let mut XUT_intermediate_data = ArrayTrait::<FixedType>::new();
//         i = 0;
//         loop {
//             if i == n {
//                 break ();
//             }
//             let mut j = 0; 
//             loop {
//                 if j == n {
//                     break ();
//                 }
//                 if i == max_row {
//                     XUT_intermediate_data.append(X.at(indices: array![row, j].span()));
//                 }
//                 else if i == row {
//                     XUT_intermediate_data.append(X.at(indices: array![max_row, j].span()));
//                 }
//                 else {
//                     XUT_intermediate_data.append(X.at(indices: array![i, j].span()));
//                 }
//                 j += 1;
//             };
//             i += 1;
//         };
//         let mut XUT_intermediate = TensorTrait::<FixedType>::new(X.shape, XUT_intermediate_data.span(), Option::Some(extra));

//         // Rearrange the y vector
//         let mut y_out_intermediate_data = ArrayTrait::<FixedType>::new();
//         i = 0;
//         loop {
//             if i == n {
//                 break ();
//             }   
//             if i == max_row {
//                 y_out_intermediate_data.append(y.at(indices: array![row].span()));
//             }
//             else if i == row {
//                 y_out_intermediate_data.append(y.at(indices: array![max_row].span()));
//             }
//             else {
//                 y_out_intermediate_data.append(y.at(indices: array![i].span()));
//             }
//             i += 1;
//         };
//         let mut y_out_intermediate = TensorTrait::<FixedType>::new(y.shape, y_out_intermediate_data.span(), Option::Some(extra));

//         // Check for singularity
//         assert(XUT_intermediate.at(indices: array![row, row].span()).mag != 0, 'matrix is singular');

//         // test_tensor(XUT_intermediate);
//         // test_tensor(y_out_intermediate);

//         // Remove zeros below diagonal
//         let mut XUT_final_data = ArrayTrait::<FixedType>::new();
//         let mut y_out_final_data = ArrayTrait::<FixedType>::new();
//         i = 0;
//         loop {
//             if i == n {
//                 // 'len(XUT):'.print();
//                 // XUT_final_data.len().print();
//                 break ();
//             }
//             else if i >= row + 1 {
//                 let mut factor = XUT_intermediate.at(indices: array![i, row].span()) / XUT_intermediate.at(indices: array![row, row].span());
//                 let mut j = 0;
//                 loop {
//                     if j == n {
//                         break ();
//                     }
//                     if j >= row {
//                         let mut val = XUT_intermediate.at(indices: array![i, j].span()) - (factor * XUT_intermediate.at(indices: array![row, j].span()));
//                         XUT_final_data.append(val);
//                     }
//                     else {
//                         XUT_final_data.append(XUT_intermediate.at(indices: array![i, j].span()));
//                     }
//                     j += 1;
//                 };
//                 let mut val_y = y_out_intermediate.at(indices: array![i].span()) - (factor * y_out_intermediate.at(indices: array![row].span()));
//                 y_out_final_data.append(val_y);
//             }
//             else {
//                 let mut j = 0;
//                 loop {
//                     if j == n {
//                         break ();
//                     }
//                     XUT_final_data.append(XUT_intermediate.at(indices: array![i, j].span()));
//                     j += 1;
//                 };
//                 y_out_final_data.append(y_out_intermediate.at(indices: array![i].span()));
//             }
//             i += 1;
//         };
//         XUT = TensorTrait::<FixedType>::new(X.shape, XUT_final_data.span(), Option::Some(extra));
//         y_out = TensorTrait::<FixedType>::new(y.shape, y_out_final_data.span(), Option::Some(extra));

//         Xy = (XUT, y_out);
        
//         row += 1;
//     };

//     return Xy;
// }

// fn back_substitution(X: Tensor::<FixedType>, y: Tensor::<FixedType>) -> Tensor::<FixedType> {
//     // Param X (Tensor::<FixedType>): 2D tensor
//     // Param y (Tensor::<FixedType>): 1D tensor
//     // Return sln (Tensor::<FixedType>): Uses back substitution to solve the system for the upper triangular matrix

//     // Initialize a tensor of zeros that will store the solutions to the system
//     let n = *X.shape.at(0);
//     let l = *y.shape.at(0);
//     assert(n == l, 'num_rows(X) != len(y)');
//     let extra = ExtraParams {fixed_point: Option::Some(FixedImpl::FP16x16(()))};

//     // Initialize sln vector
//     let mut sln_data_initial = ArrayTrait::<FixedType>::new();
//     let mut c = 0;
//     loop {
//         if c == n {
//             break ();
//         }
//         sln_data_initial.append(FixedTrait::new_unscaled(0, false));
//         c += 1;
//     };
//     let mut sln = TensorTrait::<FixedType>::new(array![n].span(), sln_data_initial.span(), Option::Some(extra));

//     // Backward iteration
//     let mut sln_data = ArrayTrait::<FixedType>::new();
//     c = n + 1;
//     loop {
//         if c == 0 {
//             break ();
//         }
//         let mut i = c - 1;

//         // Begin by assuming the solution sln[i] for the current row is the corresponding value in the vector y
//         let mut sln_i = y.at(indices: array![i].span());

//         // Iterate over columns to the right of the diagonal for the current row
//         let mut j = i + 1;
//         loop {
//             if j == n {
//                 break ();
//             }
//             // Subtract the product of the known solution sln[j] and the matrix coefficient from the current solution sln[i]
//             // How can we access previous solutions in the middle of the loop???
//             let mut val = X.at(indices: array![i, j].span()) * *sln_data.at(j - (i + 1));
//             sln_i -= val;
//         };

//         // Normalize to get the actual value
//         sln_i /= X.at(indices: array![i, i].span());
//         sln_data.append(sln_i);
        
//         c -= 1;
//     };

//     // Return the solution
//     return sln;
// }