#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "naive_blas.hpp"

std::vector<std::vector<float>> read_csv()
{
    std::string line;
    std::ifstream myfile("data/mnist_train.csv");
    std::vector<std::vector<float>> data;
    if (myfile.is_open())
    {
        getline(myfile, line);
        while (getline(myfile, line))
        {
            std::stringstream ss(line);
            std::vector<float> row;
            while (ss.good())
            {
                std::string substr;
                getline(ss, substr, ',');
                row.push_back(std::stof(substr));
            }
            data.push_back(row);
        }
        myfile.close();
    }
    return data;
}

void save_bin(int n, int m, std::vector<std::vector<float>> &data)
{
    std::vector<float> out;
    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < n; i++)
        {
            out.push_back(data[i][j]);
        }
    }

    std::ofstream outfile("data/train.bin", std::ios::out | std::ios::binary);
    if (outfile.is_open())
    {
        outfile.write(reinterpret_cast<const char *>(out.data()), sizeof(float) * n * m);
        outfile.close();
    }
    else
    {
        std::cerr << "Failed to open file: data/train.bin" << std::endl;
        throw 1;
    }
    data.clear();
}

template <typename T>
void print_vector(std::vector<T> vec)
{
    std::cout << "[";
    for (auto e : vec)
    {
        std::cout << e << ", ";
    }
    std::cout << "\b\b]\n";
}

void read_bin(const int &n, const int &m, std::vector<float> &out)
{
    // Read
    std::ifstream input("data/train.bin", std::ios::binary);
    if (!input)
    {
        std::vector<std::vector<float>> tmp = read_csv();
        save_bin(n, m, tmp);
    }

    input.read(reinterpret_cast<char *>(out.data()), sizeof(float) * n * m);
}

void print_digit(std::vector<float> &sample)
{
    std::cout << "label: " << sample[0] << "\n";
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            std::cout << std::setw(3) << sample[1 + j + i * 28] << " ";
        }
        std::cout << "\n";
    }
}

int main()
{
    // Read train data in column major order
    const int n = 60000, m = 785;
    std::vector<std::vector<float>> samples(n, std::vector<float>(m, 0));
    {
        std::vector<float> data(n * m);
        read_bin(n, m, data);
        std::cout << "shape: (" << n << ", " << m << ") size:"
                  << data.size() << " \n";
        for (int i; i < n; i++)
        {
            for (int j; j < m; j++)
            {
                samples[i][j] = data[i + j * n];
            }
        }
    }
    for (int i = 0; i < 10; i++)
    {
        print_digit(samples[i]);
    }
}