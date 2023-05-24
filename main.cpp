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

// dataloader
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

void load_data(int n, int m, std::vector<std::vector<float>> &samples)
{
    std::vector<float> data(60000 * 785);
    read_bin(60000, 785, data);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < 785; j++)
        {
            samples[i][j] = data[i + j * 60000];
            if (j > 0)
            {
                samples[i][j] /= 255;
            }
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

std::vector<float> onehot_encode(int y)
{
    std::vector<float> out(10, 0);
    out[y] = 1.f;
    return out;
}

// utils
template <typename T>
void print_vector(std::vector<T> vec)
{
    if (vec.size() == 0)
    {
        std::cout << "No element in vec\n";
        return;
    }

    std::cout << "[";
    for (auto e : vec)
    {
        std::cout << e << ", ";
    }
    std::cout << "\b\b]\n";
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

// operations
std::vector<float> log_softmax(std::vector<float> x)
{
    std::vector<float> out(x.size(), 0);
    const int n = x.size();
    float expsum = 0;
    for (int i = 0; i < n; i++)
    {
        expsum += exp(x[i]);
    }

    for (int i = 0; i < n; i++)
    {
        out[i] = x[i] - log(expsum);
    }

    return out;
}

std::vector<float> softmax(std::vector<float> x)
{
    std::vector<float> out(x.size(), 0);
    const int n = x.size();
    float expsum = 0;
    for (int i = 0; i < n; i++)
    {
        expsum += exp(x[i]);
    }

    for (int i = 0; i < n; i++)
    {
        out[i] = exp(x[i]) / expsum;
    }

    return out;
}

float cross_entropy(std::vector<float> x, std::vector<float> y)
{
    float out = 0;
    const int n = x.size();
    for (int i = 0; i < n; i++)
    {
        out -= y[i] * log(x[i]);
    }

    return out;
}

std::vector<float> grad_z_L(std::vector<float> s, std::vector<float> y)
{
    std::vector<float> out(y.size(), 0);
    for (int i = 0; i < out.size(); i++)
    {
        out[i] = s[i] - y[i];
    }

    return out;
}

// max(0, x) elementwise
void relu(std::vector<float> &x)
{
    for (int i = 0; i < x.size(); i++)
    {
        if (x[i] < 0.f)
        {
            x[i] = 0;
        }
    }
}

std::vector<float> drelu(std::vector<float> &x)
{
    std::vector<float> out(x.size(), 0);
    for (int i = 0; i < x.size(); i++)
    {
        if (x[i] > 0)
        {
            out[i] = x[i];
        }
    }

    return out;
}

class Network
{
public:
    int dim_in;
    int dim_out;
    std::vector<float> _x;
    std::vector<std::vector<float>> W;
    std::vector<float> _a;
    std::vector<float> _z;

    Network(int input_dim, int output_dim) : dim_in(input_dim), dim_out(output_dim), W()
    {
        W.resize(dim_out);
        for (int i = 0; i < dim_out; i++)
        {
            W[i].resize(dim_in + 1); // include bias
        }

        std::cout << "W shape(" << W.size() << ", " << W[0].size() << ")\n";
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

        for (int i = 0; i < W.size(); i++)
        {
            for (int j = 0; j < W[0].size(); j++)
            {
                W[i][j] = dist(rng);
            }
        }
    }

    std::vector<float> forward(std::vector<float> x)
    {
        std::vector<float> out(dim_out, 0.f);
        _x = x;
        gemv(1.f, W, x, 0.f, out);
        _a = out;
        out = softmax(out);
        _z = out;

        return out;
    }

    std::vector<std::vector<float>> backward(std::vector<float> grad)
    {
        std::vector<std::vector<float>> _grad(10, std::vector<float>(785, 0));
        for (int i = 0; i < grad.size(); i++)
        {
            for (int j = 0; j < _x.size(); j++)
            {
                _grad[i][j] = grad[i] * _x[j];
            }
        }

        return _grad;
    }
};

int main()
{
    // Read train data in column major order
    const int n = 10000, m = 785;
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

    // std::cout << y[0] << " ";
    std::vector<float> y = extract_label(samples);
    std::vector<float> y_0 = onehot_encode(y[0]);
    // print_vector(y_0);

    // print_vector(y);

    std::vector<std::vector<float>> I(10, std::vector<float>(10, 0.f));
    for (int i = 0; i < 10; i++)
    {
        I[i][i] = 1.f;
    }

    Network net(784, 10);
    net.xavier_init();

    const int MINI_BATCH_SIZE = 1000;
    std::vector<float> out, grad, y_i, x_i;
    std::vector<std::vector<float>> _grad;
    float loss, loss_mean;
    float lr = 0.005;

    int sp = 0, idx = 0;
    for (int b = 0; b < 10; b++)
    {
        loss_mean = 0;
        std::vector<std::vector<float>> _grad_mean(10, std::vector<float>(785, 0));

        for (int i = 0; i < MINI_BATCH_SIZE; i++)
        {
            idx = b * MINI_BATCH_SIZE + i;
            y_i = onehot_encode(y[idx]);
            x_i = samples[idx];
            out = net.forward(x_i);
            loss = cross_entropy(out, y_i);
            loss_mean += loss;

            grad = grad_z_L(out, y_i);
            _grad = net.backward(grad);
            gemm(1.f / MINI_BATCH_SIZE, I, _grad, 1.f, _grad_mean);
        }

        loss_mean /= MINI_BATCH_SIZE;
        gemm(lr, I, _grad_mean, 1.f, net.W);
        std::cout << "mean loss: " << loss_mean << std::endl;
    }
}