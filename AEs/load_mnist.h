#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <set>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Lê um arquivo IDX de imagens
Matrix load_images(const std::string& path, int max_images = 60000) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Erro ao abrir " + path);

    int32_t magic, num, rows, cols;
    file.read((char*)&magic, 4);
    file.read((char*)&num, 4);
    file.read((char*)&rows, 4);
    file.read((char*)&cols, 4);
    magic = __builtin_bswap32(magic);
    num = __builtin_bswap32(num);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    int n = std::min(num, max_images);
    Matrix images(n, Vector(rows * cols));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel;
            file.read((char*)&pixel, sizeof(pixel));
            double val = pixel / 255.0;
            images[i][j] = (val >= 0.5) ? 1.0 : 0.0;
        }
    }
    return images;
}
