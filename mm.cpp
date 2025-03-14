#include <vector>
#include <iostream>

typedef std::vector<unsigned> zero_nonzero;
struct Sparse {
  std::vector<zero_nonzero> rows;
  std::vector<double> values;
  unsigned N() const { return rows.size(); }
};

// Optimized Sparse matrix-vector multiplication
void sparse_mm(int N, const Sparse& A, const double* X, double* Y) {
  std::fill(Y, Y + N, 0.0);  // Efficiently initialize Y to zero
  unsigned value_index = 0;

  for (unsigned i = 0; i < N; ++i) {
    const auto& row = A.rows[i];
    unsigned col = 0;
    for (size_t j = 0; j < row.size(); j += 2) {  // Process two elements at a time
      col += row[j];  // Skip zero elements
      unsigned num_values = row[j + 1];  // Nonzero values count

      for (unsigned k = 0; k < num_values; ++k) {
        Y[i] += A.values[value_index] * X[col];
        ++value_index;
        ++col;
      }
    }
  }
}

// Optimized Dense matrix-vector multiplication
void mm(int N, const double* A, const double* X, double* Y) {
  std::fill(Y, Y + N, 0.0);  // More efficient initialization

  for (int i = 0; i < N; ++i) {
    const double* A_row = A + i * N;  // Pointer to the start of row i
    double sum = 0.0;

    for (int j = 0; j < N; ++j) {
      sum += A_row[j] * X[j];  // Optimized row-major access
    }

    Y[i] = sum;
  }
}
