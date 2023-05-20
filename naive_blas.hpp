#ifndef NAIVE_BLAS_HPP
#define NAIVE_BLAS_HPP

#include <vector>

template <typename T>
void axpy(T a, const std::vector<T> &x, std::vector<T> &y);

template <typename T>
void gemv(T a, const std::vector<std::vector<T>> &A,
          const std::vector<T> &x, T b, std::vector<T> &y);

template <typename T>
void gemm(T a, const std::vector<std::vector<T>> &A,
          const std::vector<std::vector<T>> &B,
          T b, std::vector<std::vector<T>> &C);

template <typename T>
T sdot(const std::vector<T> &x, std::vector<T> &y);

#endif // NAIVE_BLAS_HPP
