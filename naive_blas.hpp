#ifndef NAIVE_BLAS_HPP
#define NAIVE_BLAS_HPP

#include <vector>

template <typename T>
void axpy(T a, const std::vector<T> &x, std::vector<T> &y);

template <typename T>
T sdot(const std::vector<T> &x, std::vector<T> &y);

#endif // NAIVE_BLAS_HPP
