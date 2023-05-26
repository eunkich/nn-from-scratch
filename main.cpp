#include <iostream>       // std::cout
#include <vector>         // std::vector
#include <random>         // std::random_device, std::mt19337_64, std::normal_distribution
#include <cmath>          // log, sqrt
#include "naive_blas.hpp" // gemv
#include "utils.hpp"      // load_data, extract_label

// operations
std::vector<float> softmax(std::vector<float> x)
{
    std::vector<float> out(x.size(), 0.f);
    float max = x[0];
    for (auto e : x)
    {
        (e > max) ? max = e : true;
    }

    const int n = x.size();
    float expsum = 0.f;
    for (int i = 0; i < n; i++)
    {
        expsum += exp(x[i] - max);
    }

    for (int i = 0; i < n; i++)
    {
        out[i] = exp(x[i] - max) / expsum;
    }

    return out;
}

float cross_entropy(std::vector<float> x, std::vector<float> y)
{
    assert(x.size() == y.size());

    float out = 0.f;
    for (int i = 0; i < x.size(); i++)
    {
        out -= y[i] * log(x[i]);
    }
    return out;
}

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

int argmax(std::vector<float> &vec)
{
    float max = vec[0];
    int out = 0;
    for (int i = 0; i < vec.size(); i++)
    {
        if (vec[i] > max)
        {
            max = vec[i];
            out = i;
        }
    }

    return out;
}

std::vector<float> drelu(std::vector<float> &x)
{
    std::vector<float> out(x.size(), 0.f);
    for (int i = 0; i < x.size(); i++)
    {
        if (x[i] > 0)
        {
            out[i] = x[i];
        }
    }

    return out;
}

void xavier_init(std::vector<std::vector<float>> &W)
{
    int seed = 12;
    // std::random_device rd;
    // seed = rd();
    std::mt19937_64 rng;
    rng.seed(seed);
    int dim_in = W[0].size();
    int dim_out = W.size();
    float sd = sqrt(2.f / (dim_in + dim_out));
    std::normal_distribution<float> dist(0.f, sd);

    for (int i = 0; i < W.size(); i++)
    {
        for (int j = 0; j < W[0].size(); j++)
        {
            W[i][j] = dist(rng);
        }
    }
}

std::vector<float> forward(std::vector<std::vector<float>> W,
                           std::vector<float> x)
{
    std::vector<float> z(10, 0);
    gemv(1.f, W, x, 0.f, z);
    std::vector<float> a = softmax(z);

    return a;
}

std::vector<std::vector<float>> backward(std::vector<float> x,
                                         std::vector<float> y,
                                         std::vector<float> &a)
{
    std::vector<float> dz = a;
    axpy(-1.f, y, dz);
    std::vector<std::vector<float>> dW(10, std::vector<float>(785, 0.f));
    std::vector<std::vector<float>> dZ(10, std::vector<float>(1, 0.f));
    std::vector<std::vector<float>> X(1, std::vector<float>(785, 0.f));

    // expand_dims dz(10,) -> dZ(10, 1)
    for (int i = 0; i < 10; i++)
    {
        dZ[i][0] = dz[i];
    }

    // expand_dims x(785,) -> dZ(1, 785)
    for (int i = 0; i < 785; i++)
    {
        X[0][i] = x[i];
    }

    gemm(1.f, dZ, X, 0.f, dW);

    return dW;
}

int main()
{
    // Read train data in column major order
    const int n = 60000, m = 785;
    std::vector<std::vector<float>> train_X(n, std::vector<float>(m, 0.f));
    load_data(n, m, train_X);

    std::cout << "sample shape: ("
              << train_X.size() << ", "
              << train_X[0].size() << ")\n";

    std::vector<std::vector<float>> train_y = extract_label(train_X);

    // 10x10 Identity matrix
    std::vector<std::vector<float>> I(10, std::vector<float>(10, 0.f));
    for (int i = 0; i < 10; i++)
    {
        I[i][i] = 1.f;
    }

    std::vector<std::vector<float>> W(10, std::vector<float>(785, 0.f));
    xavier_init(W);

    int pred, trg;
    const int BATCH_SIZE = 64;
    const int NUM_BATCH = train_X.size() / BATCH_SIZE;
    const float LR = 0.01f;
    float mean_loss, mean_acc;
    std::vector<float> x, y, a;
    std::vector<std::vector<float>> dW(10, std::vector<float>(785, 0.f));

    int idx = 0;
    for (int epoch = 0; epoch < 10; epoch++)
    {
        mean_loss = 0.f;
        mean_acc = 0.f;

        for (int b = 0; b < NUM_BATCH; b++)
        {
            std::vector<std::vector<float>> dW_sum(10, std::vector<float>(785, 0.f));
            for (int i = 0; i < BATCH_SIZE; i++)
            {
                idx = b * BATCH_SIZE + i;
                y = train_y[idx];
                x = train_X[idx];
                a = forward(W, x);
                pred = argmax(a);
                trg = argmax(y);
                mean_acc += int(pred == trg);
                mean_loss += cross_entropy(a, y) / BATCH_SIZE;
                dW = backward(x, y, a);
                gemm(1.f, I, dW, 1.f, dW_sum);
            }
            gemm(-LR / BATCH_SIZE, I, dW_sum, 1.f, W);
        }

        std::cout << "epoch: " << epoch + 1 << "/10 "
                  << "mean loss: " << mean_loss / (NUM_BATCH) << " "
                  << "mean acc: " << mean_acc / (NUM_BATCH * BATCH_SIZE) << "\n";
    }
}