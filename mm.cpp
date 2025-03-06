#include <vector>

// A NxN sparse representation
typedef std::vector<unsigned> zero_nonzero;
struct Sparse {
  std::vector<zero_nonzero> rows;  // form: zeros,non,zeros,non...
  std::vector<double> values;
  unsigned N() const { return rows.size(); }
};

// X and Y are both aligned on 4096 byte boundaries
void sparse_mm(int N, const Sparse& A, const double* X, double* Y) {
  for (unsigned i = 0; i < A.N(); ++i) {
    Y[i] = 0.0;  // Initialize the output vector element to zero
    unsigned value_index = 0;
    unsigned col = 0;
    for (unsigned j = 0; j < A.rows[i].size(); ++j) {
      unsigned zeros = A.rows[i][j];
      col += zeros;  // Skip zeros
      if (j % 2 == 1) {  // Non-zero count
        unsigned non_zeros = A.rows[i][j];
        for (unsigned k = 0; k < non_zeros; ++k) {
          Y[i] += A.values[value_index++] * X[col++];
        }
      } else {
        col += zeros;  // Skip zeros
      }
    }
  }
}

// A, X, and Y are all aligned on 4096 byte boundaries
void mm(int N, const double* A, const double* X, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = 0.0;  // Initialize the output vector element to zero
    for (int j = 0; j < N; ++j) {
      Y[i] += A[i * N + j] * X[j];
    }
  }
}
