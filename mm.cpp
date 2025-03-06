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
    for (unsigned j = 0; j < row.size(); j += 2) {
      // Skip zeros
      col += row[j];
      // Process non-zeros
      unsigned non_zeros = row[j + 1];
      for (unsigned k = 0; k < non_zeros; ++k) {
        Y[i] += A.values[value_index++] * X[col++];
      }
    }
  }
}

// Dense matrix-vector multiplication
void mm(int N, const double* A, const double* X, double* Y) {
  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    double sum = 0.0;
    const double* A_row = A + i * N;  // Pointer to the start of the current row
    for (int j = 0; j < N; ++j) {
      sum += A_row[j] * X[j];
    }
    Y[i] = sum;
  }
}
