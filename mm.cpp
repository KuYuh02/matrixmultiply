#include <vector>
#include <iostream>
#include <cstring>
#include <omp.h>

// Sparse matrix representation for an NxN matrix
typedef std::vector<unsigned> zero_nonzero;
struct Sparse {
    std::vector<zero_nonzero> rows;  // Alternating zero and nonzero counts
    std::vector<double> values;      // Nonzero matrix values
    unsigned N() const { return rows.size(); }
};

// Sparse matrix-vector multiplication
void sparse_mm(int N, const Sparse& A, const double* X, double* Y) {
    std::memset(Y, 0, N * sizeof(double));  // Zero out the result vector Y
    size_t value_index = 0;

    // Temporary array for parallel result accumulation
    std::vector<double> tempY(N, 0.0);

    // Parallelize row-wise computation
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        const zero_nonzero& row = A.rows[i];
        size_t col_index = 0;

        // Iterate through row segments (zero and nonzero)
        for (size_t j = 0; j < row.size(); ++j) {
            if (j % 2 == 1) { // Nonzero segment
                for (unsigned k = 0; k < row[j]; ++k) {
                    tempY[i] += A.values[value_index++] * X[col_index++];
                }
            } else { // Zero segment
                col_index += row[j]; // Skip over zero elements
            }
        }
    }

    // Transfer accumulated results into the output vector Y
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        Y[i] = tempY[i];
    }
}

// Dense matrix-vector multiplication
void mm(int N, const double* A, const double* X, double* Y) {
    std::memset(Y, 0, N * sizeof(double));  // Zero out the result vector Y

    // Block size for improved cache efficiency
    const int block_size = 16;

    // Blocked matrix multiplication
    for (int i = 0; i < N; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int ii = i; ii < std::min(i + block_size, N); ++ii) {
                for (int jj = j; jj < std::min(j + block_size, N); ++jj) {
                    Y[ii] += A[ii * N + jj] * X[jj];
                }
            }
        }
    }
}
