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

  // Example vector X
  double X[N] = {1, 2, 3, 4};

  // Output vectors
  double Y_sparse[N] = {0};
  double Y_dense[N] = {0};

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
