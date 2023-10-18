/*
Gautam Ahuja
Introduction to Machine Learning
Raghavendra Singh
Assignment 3
Implement PCA using Eigen library
*/

#include <eigen/Eigen/Eigenvalues>
#include <eigen/Eigen/Dense>
#include <eigen/Eigen>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

using namespace Eigen;

void printMatrix(MatrixXd m){
    rows = m.rows();
    cols = m.cols();
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            printf("%lf ", m(i,j));
        }
        printf("\n");
    }
}

// Function for PCA
// Input: X (m x n) matrix
// Output: Y (m x k) matrix
// k = number of principal components
MatrixXd PCA(MatrixXd X, int n_components) { 
    // Calculate covariance matrix -- (X^T * X) / (m-1)
    MatrixXd X_T = X.transpose();
    MatrixXd cov = (X_T * X) / (X.rows()-1);
    printf("Covariance Matrix: \n");
    printMatrix(cov);

    // Calculate eigenvalues and eigenvectors -- using Eigen library
    SelfAdjointEigenSolver<MatrixXd> es(cov);
    MatrixXd eigenvalues = es.eigenvalues();
    MatrixXd eigenvectors = es.eigenvectors();
    printf("Eigenvalues: \n");
    printMatrix(eigenvalues);
    printf("Eigenvectors: \n");

    // Calculate the variance explained by each principal component -- (eigenvalue / sum of eigenvalues) * 100
    MatrixXd variance_explained = (eigenvalues / eigenvalues.sum()) * 100;
    printf("Variance Explained: \n");
    printMatrix(variance_explained);

    // Standardize the data -- (X - X.mean()) 
    MatrixXd X_std = (X - X.mean()); // / X.std();

    // Sort the eigenvalues in descending order -- using Eigen library
    MatrixXd sorted_eigenvalues = eigenvalues.colwise().reverse();
    MatrixXd sorted_eigenvectors = eigenvectors.rowwise().reverse();

    // Compute the projection matrix -- first n_components columns of sorted_eigenvectors
    MatrixXd projection_matrix = sorted_eigenvectors.block(0,0,sorted_eigenvectors.rows(),n_components);
    printf("Projection Matrix: \n");
    printMatrix(projection_matrix);

    // Project the data onto the lower-dimensional subspace -- X_std * projection_matrix
    MatrixXd Y = X_std * projection_matrix;
    printf("Y: \n");
    printMatrix(Y);
    
    // Return new matrix Y. Its data type is MatrixXd
    return Y;
}

int main(){
    // Generate random datasets with (m, n ) = [(100,100), (1000,1000), (10000, 10000), (10000,50000), (50000,50000)]
    // m = number of samples
    // n = number of features
    // n_components = number of principal components = 0.1*n = [10, 100, 1000, 5000, 5000]
    
    /*************************** Matrix 1: m = 100, n = 100 ***************************/
    int m = 100;
    int n = 100;
    int n_components = 0.1 * n;
    // Generate random matrix
    MatrixXd X1 = MatrixXd::Random(m,n);
    // Call PCA function
    int start_time = clock();
    MatrixXd Y1 = PCA(X1, n_components);   
    int end_time = clock();
    printf("Time taken for PCA: %d\n", end_time - start_time);

    /*************************** Matrix 2: m = 1000, n = 1000 ***************************/
    m = 1000;
    n = 1000;
    n_components = 0.1 * n;
    // Generate random matrix
    MatrixXd X2 = MatrixXd::Random(m,n);
    // Call PCA function
    start_time = clock();
    MatrixXd Y2 = PCA(X2, n_components);
    end_time = clock();
    printf("Time taken for PCA: %d\n", end_time - start_time);

    /*************************** Matrix 3: m = 10000, n = 10000 ***************************/
    m = 10000;
    n = 10000;
    n_components = 0.1 * n;
    // Generate random matrix
    MatrixXd X3 = MatrixXd::Random(m,n);
    // Call PCA function
    start_time = clock();
    MatrixXd Y3 = PCA(X3, n_components);
    end_time = clock();
    printf("Time taken for PCA: %d\n", end_time - start_time);

    /*************************** Matrix 4: m = 10000, n = 50000 ***************************/
    m = 10000;
    n = 50000;
    n_components = 0.1 * n;
    // Generate random matrix
    MatrixXd X4 = MatrixXd::Random(m,n);
    // Call PCA function
    start_time = clock();
    MatrixXd Y4 = PCA(X4, n_components);
    end_time = clock();
    printf("Time taken for PCA: %d\n", end_time - start_time);

    /*************************** Matrix 5: m = 50000, n = 50000 ***************************/
    m = 50000;
    n = 50000;
    n_components = 0.1 * n;
    // Generate random matrix
    MatrixXd X5 = MatrixXd::Random(m,n);
    // Call PCA function
    start_time = clock();
    MatrixXd Y5 = PCA(X5 n_components);
    end_time = clock();
    printf("Time taken for PCA: %d\n", end_time - start_time);

    return 0;
}