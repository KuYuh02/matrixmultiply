#include <vector>
#include <iostream>

typedef std::vector<unsigned> zero_nonzero;
struct Sparse {
  std::vector<zero_nonzero> rows;
  std::vector<double> values;
  unsigned N() const { return rows.size(); }
};

// Sparse matrix-vector multiplication
void sparse_mm(int N, const Sparse& A, const double* X, double* Y) {
  unsigned value_index = 0;
  for (unsigned i = 0; i < N; ++i) {
    Y[i] = 0.0;
    const auto& row = A.rows[i];
    unsigned col = 0;
    for (unsigned j = 0; j < row.size(); ++j) {
      if (j % 2 == 0) {
        col += row[j];  // Skip zero elements
      } else {
        for (unsigned k = 0; k < row[j]; ++k) {
          Y[i] += A.values[value_index] * X[col];
          ++value_index;
          ++col;
        }
      }
    }
  }
}

// Dense matrix-vector multiplication
void mm(int N, const double* A, const double* X, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = 0.0;
    for (int j = 0; j < N; ++j) {
      Y[i] += A[i * N + j] * X[j];
    }
  }
}
