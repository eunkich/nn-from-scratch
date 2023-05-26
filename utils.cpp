#include <iostream>  // std::cout
#include <iomanip>   // std::setw
#include <fstream>   // std::ifstream
#include <sstream>   // std::stringstream
#include <vector>    // std::vector
#include <string>    // std::string
#include <algorithm> // std::swap

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

std::vector<float> onehot_encode(int y)
{
    std::vector<float> out(10, 0.f);
    out[y] = 1.f;
    return out;
}

// extract labels and replace it with 1 to use it for bias
// samples[i]: [y, x...x] -> [1, x...x]
// y[i]: label for samples[i]
// return vector of onehot encoded lables after swap
std::vector<std::vector<float>> extract_label(std::vector<std::vector<float>> &samples)
{
    std::vector<float> y(samples.size(), 1.f);
    std::vector<std::vector<float>> out;
    for (int i = 0; i < y.size(); i++)
    {
        std::swap(y[i], samples[i][0]);
        out.push_back(onehot_encode(y[i]));
    }

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
            (sample[1 + j + i * 28] != 0) ? std::cout << "1" : std::cout << " ";
            std::cout << " ";
        }
        std::cout << "\n";
    }
}

template void print_vector(std::vector<float> vec);