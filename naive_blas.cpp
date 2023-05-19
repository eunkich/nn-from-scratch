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

template float sdot(const std::vector<float> &x, std::vector<float> &y);
template void axpy(float a, const std::vector<float> &x, std::vector<float> &y);
