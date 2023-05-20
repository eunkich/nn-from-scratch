#include <iostream>
#include <random>

#include "naive_blas.hpp"

template <typename T>
void axpy(T a, const std::vector<T> &x, std::vector<T> &y)
{
    if (x.size() != y.size())
    {
        std::cerr << "Error: vectors have different sizes" << std::endl;
        return;
    }

    for (int i = 0; i < x.size(); i++)
    {
        y[i] += a * x[i];
    }
}

template <typename T>
void gemm(T a, const std::vector<std::vector<T>> &A,
          const std::vector<std::vector<T>> &B,
          T b, std::vector<std::vector<T>> &C)
{
    int m = A.size();
    int p = A[0].size();
    int n = B[0].size();
    if (B.size() != p)
    {
        std::cerr << "Error: matrix dimensions A(m x p) and B(p x n) are not compatible" << std::endl;
        return;
    }
    if (C.size() != m or C[0].size() != n)
    {
        std::cerr << "Error: matrix dimensions AB(m x n) and C(m x n) are not compatible" << std::endl;
        return;
    }

    T tmp;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            tmp = 0;

            for (int k = 0; k < p; k++)
            {
                tmp += A[i][k] * B[k][j];
            }

            tmp *= a;
            C[i][j] *= b;
            C[i][j] += a * tmp;
        }
    }
}
template <typename T>
void gemv(T a, const std::vector<std::vector<T>> &A,
          const std::vector<T> &x, T b, std::vector<T> &y)
{
    if (x.size() != A.size())
    {
        std::cerr << "Error: matrix A(m x n) and vector x(n x 1) have different sizes" << std::endl;
        return;
    }
    if (y.size() != A[0].size())
    {
        std::cerr << "Error: matrix Ax(m x 1) and vector y(m x 1) have different sizes" << std::endl;
        return;
    }

    for (int i = 0; i < A.size(); i++)
    {
        y[i] *= b;

        T tmp = 0;
        for (int j = 0; j < x.size(); j++)
        {
            tmp += A[i][j] * x[j];
        }

        y[i] += a * tmp;
    }
}

template <typename T>
T sdot(const std::vector<T> &x, std::vector<T> &y)
{
    if (x.size() != y.size())
    {
        std::cerr << "Error: vectors have different sizes" << std::endl;
        throw std::runtime_error("error");
    }
    T out = 0;
    for (int i = 0; i < x.size(); i++)
    {
        out += x[i] * y[i];
    }

    return out;
}

template void axpy(float a, const std::vector<float> &x, std::vector<float> &y);
template void gemv(float a, const std::vector<std::vector<float>> &A,
                   const std::vector<float> &x, float b, std::vector<float> &y);
template void gemm(float a, const std::vector<std::vector<float>> &A,
                   const std::vector<std::vector<float>> &B,
                   float b, std::vector<std::vector<float>> &C);
template float sdot(const std::vector<float> &x, std::vector<float> &y);
