#include <iostream>       // std::cout
#include <iomanip>        // std::setw
#include <fstream>        // std::ifstream
#include <sstream>        // std::stringstream
#include <string>         // std::string
#include <vector>         // std::vector
#include <random>         // std::random_device, std::mt19337_64, std::normal_distribution
#include <algorithm>      // std::swap
#include <math.h>         // log, sqrt
#include "naive_blas.hpp" // gemv

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

float log_softmax(int i, std::vector<float> z)
{
    const int n = z.size();
    float expsum = 0;
    for (int i = 0; i < n; i++)
    {
        expsum += exp(z[i]);
    }

    return z[i] - log(expsum);
}

void load_data(int n, int m, std::vector<std::vector<float>> &samples)
{
    std::vector<float> data(60000 * 785);
    read_bin(60000, 785, data);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < 785; j++)
        {
            samples[i][j] = data[i + j * 60000] / 255;
        }
    }
}

// extract labels and replace it with 1 to use it for bias
// samples[i]: [y, x...x] -> [1, x...x]
// y[i]: label for samples[i]
std::vector<float> extract_label(std::vector<std::vector<float>> &samples)
{
    std::vector<float> y(samples.size(), 1);
    for (int i = 0; i < y.size(); i++)
    {
        std::swap(y[i], samples[i][0]);
    }
    return y;
}

class Network
{
public:
    int dim_in;
    int dim_out;
    std::vector<std::vector<float>> weights;
    Network(int input_dim, int output_dim) : dim_in(input_dim), dim_out(output_dim), weights()
    {
        weights.resize(dim_out);
        for (int i = 0; i < dim_out; i++)
        {
            weights[i].resize(dim_in + 1); // include bias
        }

        std::cout << "W shape(" << weights.size() << ", " << weights[0].size() << ")\n";
    }

    void xavier_init()
    {
        int seed;
        std::random_device rd;
        std::mt19937_64 rng;
        seed = rd();
        rng.seed(seed);
        float var = sqrt(2.f / static_cast<float>(dim_in + dim_out));
        std::normal_distribution<float> dist(0.f, var);

        for (int i = 0; i < weights.size(); i++)
        {
            for (int j = 0; j < weights[0].size(); j++)
            {
                weights[i][j] = dist(rng);
            }
        }
    }

    std::vector<float> forward(std::vector<float> x)
    {
        std::vector<float> y(dim_out, 0.f);
        gemv(1.f, weights, x, 0.f, y);
        return y;
    }
};

int main()
{
    // Read train data in column major order
    const int n = 1000, m = 785;
    std::vector<std::vector<float>> samples(n, std::vector<float>(m, 0));
    load_data(n, m, samples);

    std::cout << "sample shape: ("
              << samples.size() << ", "
              << samples[0].size() << ")\n";

    // head
    // for (int i = 0; i < 10; i++)
    // {
    //     print_digit(samples[i]);
    // }

    std::vector<float> y = extract_label(samples);
    // print_vector(y);

    Network net(784, 10);
    net.xavier_init();
    std::vector<float> out = net.forward(samples[0]);
    print_vector(out);
    float loss = log_softmax(y[0], out);
    std::cout << loss << "\n";
    // std::cout << net.weights.size() << net.weights[0].size() << std::endl;
}