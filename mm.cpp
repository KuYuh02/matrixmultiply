#include <vector>
#include <iostream>

typedef std::vector<unsigned> zero_nonzero;
struct Sparse {
  std::vector<zero_nonzero> rows;
  std::vector<double> values;
  unsigned N() const { return rows.size(); }
};

// Fixed and optimized Sparse matrix-vector multiplication
void sparse_mm(int N, const Sparse& A, const double* X, double* Y) {
  std::fill(Y, Y + N, 0.0);  // Efficiently zero out result vector
  unsigned value_index = 0;

  for (unsigned i = 0; i < N; ++i) {
    const auto& row = A.rows[i];
    unsigned col = 0;

    for (size_t j = 0; j < row.size(); ++j) {
      if (j % 2 == 0) {
        col += row[j];  // Skip zero elements
      } else {
        unsigned nonzeros = row[j];
        for (unsigned k = 0; k < nonzeros; ++k) {
          if (value_index >= A.values.size()) break;  // Safety check
          if (col >= N) break;  // Avoid out-of-bounds access
          Y[i] += A.values[value_index] * X[col];
          ++value_index;
          ++col;
        }
      }
    }
  }
}

// Fixed and optimized Dense matrix-vector multiplication
void mm(int N, const double* A, const double* X, double* Y) {
  std::fill(Y, Y + N, 0.0);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      Y[i] += A[i * N + j] * X[j];
    }
  }
}
