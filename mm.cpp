#include <vector>
#include <iostream>

// A NxN sparse representation
typedef std::vector<unsigned> zero_nonzero;
struct Sparse {
  std::vector<zero_nonzero> rows;  // form: zeros,non,zeros,non...
  std::vector<double> values;
  unsigned N() const { return rows.size(); }
};

// Sparse matrix-vector multiplication
void sparse_mm(int N, const Sparse& A, const double* X, double* Y) {
  // Initialize Y to zero
  for (int i = 0; i < N; ++i) {
    Y[i] = 0.0;
  }

  int value_index = 0; // Index to track the position in the values array

  // Iterate over each row in the sparse matrix
  for (int i = 0; i < N; ++i) {
    const zero_nonzero& row = A.rows[i];
    int col_index = 0; // Column index in the matrix

    // Iterate over the zero_nonzero pattern for the current row
    for (size_t j = 0; j < row.size(); ++j) {
      if (j % 2 == 0) {
        // Even index: skip zeros
        col_index += row[j];
      } else {
        // Odd index: process non-zero values
        for (unsigned k = 0; k < row[j]; ++k) {
          Y[i] += A.values[value_index++] * X[col_index++];
        }
      }
    }
  }
}

// Dense matrix-vector multiplication
void mm(int N, const double* A, const double* X, double* Y) {
  // Initialize Y to zero
  for (int i = 0; i < N; ++i) {
    Y[i] = 0.0;
  }

  // Perform matrix-vector multiplication
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      Y[i] += A[i * N + j] * X[j];
    }
  }
}

// Example usage
int main() {
  // Example sparse matrix
  Sparse A;
  A.rows = {{2, 2, 1}, {0, 3, 1, 1}, {5}, {0, 5}};
  A.values = {4, 5, 1, 2, 3, 4, 1, 1, 1, 1, 1};

  // Example dense matrix
  const int N = 4;
  double dense_A[N * N] = {
    0, 0, 4, 5,
    1, 2, 3, 0,
    0, 0, 0, 0,
    1, 1, 1, 1
  };

  // Example vector
  double X[N] = {1, 2, 3, 4};

  // Result vectors
  double Y_sparse[N];
  double Y_dense[N];

  // Perform sparse matrix-vector multiplication
  sparse_mm(N, A, X, Y_sparse);

  // Perform dense matrix-vector multiplication
  mm(N, dense_A, X, Y_dense);

  // Print results
  std::cout << "Sparse result: ";
  for (int i = 0; i < N; ++i) {
    std::cout << Y_sparse[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Dense result: ";
  for (int i = 0; i < N; ++i) {
    std::cout << Y_dense[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
