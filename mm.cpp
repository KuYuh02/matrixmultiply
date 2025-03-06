#include <vector>
#include <iostream>
#include <algorithm>
#include <omp.h>

typedef std::vector<unsigned> zero_nonzero;

struct Sparse {
  std::vector<zero_nonzero> rows;
  std::vector<double> values;
  unsigned N() const { return rows.size(); }
};

// Optimized sparse matrix-vector multiplication
void sparse_mm(int N, const Sparse& A, const double* X, double* Y) {
  // Input validation
  if (N <= 0 || !X || !Y || A.N() != static_cast<unsigned>(N)) {
    std::cerr << "Invalid input parameters to sparse_mm" << std::endl;
    return;
  }
  
  // Initialize Y to zero
  std::fill(Y, Y + N, 0.0);
  
  unsigned value_index = 0;
  
  #pragma omp parallel for if(N > 1000)
  for (int i = 0; i < N; ++i) {
    double temp_sum = 0.0;  // Use local accumulator for better cache performance
    const auto& row = A.rows[i];
    unsigned col = 0;
    
    // Process row elements in chunks for better cache locality
    for (unsigned j = 0; j < row.size(); j += 2) {
      // Process zeros (skip them)
      if (j < row.size()) {
        col += row[j];
      }
      
      // Process non-zeros
      if (j + 1 < row.size()) {
        const unsigned non_zeros = row[j + 1];
        const unsigned local_value_index = value_index + (i == 0 ? 0 : 
            std::accumulate(A.rows.begin(), A.rows.begin() + i, 0u,
                           [](unsigned acc, const zero_nonzero& r) {
                             return acc + std::accumulate(r.begin() + 1, r.end(), 0u, 
                                                         [](unsigned sum, unsigned val) {
                                                           return sum + (val * ((&val - &r[0]) % 2));
                                                         });
                           }));
        
        // Use vectorization hints for compiler
        #pragma omp simd reduction(+:temp_sum)
        for (unsigned k = 0; k < non_zeros; ++k) {
          temp_sum += A.values[local_value_index + k] * X[col + k];
        }
        
        col += non_zeros;
      }
    }
    
    Y[i] = temp_sum;
  }
}

// Optimized dense matrix-vector multiplication
void mm(int N, const double* A, const double* X, double* Y) {
  // Input validation
  if (N <= 0 || !A || !X || !Y) {
    std::cerr << "Invalid input parameters to mm" << std::endl;
    return;
  }
  
  // Initialize Y to zero
  std::fill(Y, Y + N, 0.0);
  
  // Use block-based approach for better cache performance
  const int block_size = 64;  // Adjust based on cache line size
  
  #pragma omp parallel for if(N > 1000)
  for (int i = 0; i < N; ++i) {
    double temp_sum = 0.0;
    const double* row = A + i * N;
    
    // Process in blocks for better cache utilization
    for (int jb = 0; jb < N; jb += block_size) {
      const int end_j = std::min(jb + block_size, N);
      
      // Use vectorization hints for compiler
      #pragma omp simd reduction(+:temp_sum)
      for (int j = jb; j < end_j; ++j) {
        temp_sum += row[j] * X[j];
      }
    }
    
    Y[i] = temp_sum;
  }
}

// Helper function to create a sparse matrix from a dense matrix
Sparse create_sparse_matrix(int N, const double* dense_matrix, double epsilon = 1e-10) {
  Sparse result;
  result.rows.resize(N);
  
  for (int i = 0; i < N; ++i) {
    const double* row = dense_matrix + i * N;
    unsigned consecutive_zeros = 0;
    unsigned consecutive_nonzeros = 0;
    
    for (int j = 0; j < N; ++j) {
      if (std::abs(row[j]) < epsilon) {
        // Current element is zero
        if (consecutive_nonzeros > 0) {
          // Store the previous run of non-zeros
          result.rows[i].push_back(consecutive_nonzeros);
          consecutive_nonzeros = 0;
        }
        consecutive_zeros++;
      } else {
        // Current element is non-zero
        if (consecutive_zeros > 0) {
          // Store the previous run of zeros
          result.rows[i].push_back(consecutive_zeros);
          consecutive_zeros = 0;
        }
        consecutive_nonzeros++;
        result.values.push_back(row[j]);
      }
    }
    
    // Handle any remaining runs at the end of the row
    if (consecutive_zeros > 0) {
      result.rows[i].push_back(consecutive_zeros);
    }
    if (consecutive_nonzeros > 0) {
      result.rows[i].push_back(consecutive_nonzeros);
    }
  }
  
  return result;
}
