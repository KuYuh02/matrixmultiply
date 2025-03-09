#include <vector>
#include <iostream>

// A NxN sparse representation
typedef std::vector<unsigned> zero_nonzero;
struct Sparse {
  std::vector<zero_nonzero> rows;  // form: zeros,non,zeros,non...
  std::vector<double> values;
  unsigned N() const { return rows.size(); }
};

// X and Y are both aligned on 4096 byte boundaries
void sparse_mm(int N, const Sparse& A, const double* X, double* Y) {
  for (unsigned i = 0; i < N; ++i) {
    Y[i] = 0.0;
    unsigned value_index = 0;
    unsigned offset = 0;
    for (unsigned j = 0; j < A.rows[i].size(); ++j) {
      if (j % 2 == 0) {
        offset += A.rows[i][j];
      } else {
        for (unsigned k = 0; k < A.rows[i][j]; ++k) {
          Y[i] += A.values[value_index] * X[offset];
          ++value_index;
          ++offset;
        }
      }
    }
  }
}

// A, X, and Y are all aligned on 4096 byte boundaries
void mm(int N, const double* A, const double* X, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = 0.0;
    for (int j = 0; j < N; ++j) {
      Y[i] += A[i * N + j] * X[j];
    }
  }
}
