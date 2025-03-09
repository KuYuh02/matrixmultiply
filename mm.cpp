#include <vector> 
#include <iostream> 
 
// A NxN sparse representation 
typedef std::vector<unsigned> zero_nonzero; 
struct Sparse { 
  // Invariant: len(rows) == N 
  // Invariant: sum(x for x in row[i]) == N 
  // Invariant: len(values) == sum(all non-zero counts) 
  // Invariant: rows[i][j] is a count of zeros where j is even 
  // We represent rows like this: 
  // 0 0 4 5 0    (2,2,1)    2+2+1 = 5 
  // 1 2 3 0 4    (0,3,1,1)  0+3+1+1 = 5 
  // 0 0 0 0 0    (5)        5 = 5 
  // 1 1 1 1 1    (0,5)      0+5 = 5 
  // values holds only the non-zero entries 
  // 4 5 1 2 3 4 1 1 1 1 1 
   
  std::vector<zero_nonzero> rows;  // form: zeros,non,zeros,non... 
  std::vector<double> values; 
  unsigned N() const { return rows.size(); } 
}; 
 
// X and Y are both aligned on 4096 byte boundaries 
void sparse_mm(int N, const Sparse& A, const double* X, double* Y) {
  // Initialize the result vector Y to zero
  for (int i = 0; i < N; i++) {
    Y[i] = 0.0;
  }
  
  // Value index to track current position in the values array
  size_t value_idx = 0;
  
  // Process each row of the sparse matrix
  for (int i = 0; i < N; i++) {
    const zero_nonzero& row_pattern = A.rows[i];
    int col = 0;
    
    // Process the pattern of zeros and non-zeros
    for (size_t pattern_idx = 0; pattern_idx < row_pattern.size(); pattern_idx++) {
      unsigned count = row_pattern[pattern_idx];
      
      if (pattern_idx % 2 == 0) {
        // Skip zeros
        col += count;
      } else {
        // Process non-zero elements
        for (unsigned j = 0; j < count; j++) {
          Y[i] += A.values[value_idx] * X[col];
          value_idx++;
          col++;
        }
      }
    }
  }
}
 
// A, X, and Y are all aligned on 4096 byte boundaries
void mm(int N, const double* A, const double* X, double* Y) {
  // Initialize the result vector Y to zero
  for (int i = 0; i < N; i++) {
    Y[i] = 0.0;
  }
  
  // Perform matrix-vector multiplication
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      Y[i] += A[i * N + j] * X[j];
    }
  }
}

// Example usage function
void example_usage() {
  const int N = 5;
  
  // Create a sparse matrix example
  Sparse sparse_matrix;
  
  // Row 0: 0 0 4 5 0 (2 zeros, 2 non-zeros, 1 zero)
  sparse_matrix.rows.push_back({2, 2, 1});
  sparse_matrix.values.push_back(4);
  sparse_matrix.values.push_back(5);
  
  // Row 1: 1 2 3 0 4 (0 zeros, 3 non-zeros, 1 zero, 1 non-zero)
  sparse_matrix.rows.push_back({0, 3, 1, 1});
  sparse_matrix.values.push_back(1);
  sparse_matrix.values.push_back(2);
  sparse_matrix.values.push_back(3);
  sparse_matrix.values.push_back(4);
  
  // Row 2: 0 0 0 0 0 (5 zeros)
  sparse_matrix.rows.push_back({5});
  
  // Row 3: 1 1 1 1 1 (0 zeros, 5 non-zeros)
  sparse_matrix.rows.push_back({0, 5});
  sparse_matrix.values.push_back(1);
  sparse_matrix.values.push_back(1);
  sparse_matrix.values.push_back(1);
  sparse_matrix.values.push_back(1);
  sparse_matrix.values.push_back(1);
  
  // Row 4: 0 0 0 0 0 (5 zeros)
  sparse_matrix.rows.push_back({5});
  
  // Create a sample vector X
  double X[N] = {1.0, 2.0, 3.0, 4.0, 5.0};
  double Y[N] = {0.0}; // Result vector
  
  // Perform sparse matrix-vector multiplication
  sparse_mm(N, sparse_matrix, X, Y);
  
  std::cout << "Sparse matrix-vector multiplication result:" << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "Y[" << i << "] = " << Y[i] << std::endl;
  }
  
  // Create a dense matrix example
  double A[N * N] = {
    0, 0, 4, 5, 0,
    1, 2, 3, 0, 4,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0
  };
  
  // Reset Y
  for (int i = 0; i < N; i++) {
    Y[i] = 0.0;
  }
  
  // Perform dense matrix-vector multiplication
  mm(N, A, X, Y);
  
  std::cout << "Dense matrix-vector multiplication result:" << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "Y[" << i << "] = " << Y[i] << std::endl;
  }
}
