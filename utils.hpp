#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

std::vector<std::vector<float>> read_csv();

void save_bin(int n, int m, std::vector<std::vector<float>> &data);

void read_bin(const int &n, const int &m, std::vector<float> &out);

void load_data(int n, int m, std::vector<std::vector<float>> &samples, std::string path);

std::vector<std::vector<float>> extract_label(std::vector<std::vector<float>> &samples);

std::vector<float> onehot_encode(int y);

template <typename T>
void print_vector(std::vector<T> vec);

void print_digit(std::vector<float> &sample);

#endif // UTILS_HPP
