#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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

int main()
{
    // Read
    std::ifstream input("data/train.bin", std::ios::binary);
    const int n = 60000, m = 785;
    if (!input)
    {
        std::vector<std::vector<float>> data = read_csv();
        save_bin(n, m, data);
    }

    std::vector<float> matrix_read(n * m);
    input.read(reinterpret_cast<char *>(matrix_read.data()), sizeof(float) * n * m);
    std::cout << "shape: (" << n << ", " << m << ") size:"
              << matrix_read.size() << " \n";
}