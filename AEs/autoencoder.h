#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <iomanip>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double sigmoid_derivative(double y) { return y * (1 - y); }

double binary_cross_entropy(const Vector& target, const Vector& output) {
    double loss = 0.0;
    const double eps = 1e-12;
    for (size_t k = 0; k < target.size(); ++k) {
        double y = target[k];
        double y_hat = std::clamp(output[k], eps, 1.0 - eps);
        loss += -(y * std::log(y_hat) + (1.0 - y) * std::log(1.0 - y_hat));
    }
    return loss;
}

void printProgress(double progress, int epoch, double loss) {
    int barWidth = 50;
    if (epoch == 0) {
        int pos = static_cast<int>(barWidth * progress);
        std::cout << "Progress: [";
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    } else {
        int epoch_width {3};
        std::cout << "Epoch: " << std::left << std::setw(epoch_width) << epoch << " -> Loss: " << 
        std::fixed << std::setprecision(4) << std::left << std::setw(8) << loss << " | Prog.: [";
        int pos = static_cast<int>(barWidth * progress);
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    }
}

void save_pgm(const Vector& img, const std::string& filename) {
    std::ofstream f(filename, std::ios::binary);
    f << "P5\n28 28\n255\n";
    for (double val : img) {
        unsigned char pixel = static_cast<unsigned char>(std::clamp(val * 255.0, 0.0, 255.0));
        f.write((char*)&pixel, 1);
    }
}

class Autoencoder {
public:
    Autoencoder(const std::vector<int>& topology, double lr = 0.05);
    Vector feedforward(const Vector& input);
    void backpropagate(const Vector& target);
    void updateWeights();
    void train(const Matrix& data, int epochs, double threshold);
    void test(const Matrix& data);
    void reconstruct(const Matrix& data);
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);

private:
    std::vector<int> topology;
    double learningRate;
    std::vector<Vector> outputs, deltas, biases;
    std::vector<Matrix> weights;
};

Autoencoder::Autoencoder(const std::vector<int>& topo, double lr) : topology(topo), learningRate(lr) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    for (size_t i {0}; i < topology.size(); ++i) {
        outputs.push_back(Vector(topology[i]));
        deltas.push_back(Vector(topology[i]));
        biases.push_back(Vector(topology[i]));
        if (i > 0) {
            weights.push_back(Matrix(topology[i], Vector(topology[i - 1])));
            for (int j {0}; j < topology[i]; ++j) {
                biases[i][j] = dis(gen);
                for (int k {0}; k < topology[i - 1]; ++k)
                    weights[i - 1][j][k] = dis(gen);
            }
        }
    }
}

Vector Autoencoder::feedforward(const Vector& input) {
    outputs[0] = input;
    for (size_t i {1}; i < topology.size(); ++i) {
        for (int j {0}; j < topology[i]; ++j) {
            double sum = biases[i][j];
            for (int k {0}; k < topology[i - 1]; ++k)
                sum += weights[i - 1][j][k] * outputs[i - 1][k];
            outputs[i][j] = sigmoid(sum);
        }
    }
    return outputs.back();
}

void Autoencoder::backpropagate(const Vector& target) {
    int L = topology.size() - 1;

    // Camada de saída
    for (int j {0}; j < topology[L]; ++j) {
        double error = target[j] - outputs[L][j];
        deltas[L][j] = error * sigmoid_derivative(outputs[L][j]);
    }

    // Camadas ocultas
    for (int i = L - 1; i > 0; --i) {
        for (int j {0}; j < topology[i]; ++j) {
            double errorSum {0.0};
            for (int k {0}; k < topology[i + 1]; ++k)
                errorSum += weights[i][k][j] * deltas[i + 1][k];
            deltas[i][j] = errorSum * sigmoid_derivative(outputs[i][j]);
        }
    }
}

void Autoencoder::updateWeights() {
    for (size_t i {1}; i < topology.size(); ++i) {
        for (int j {0}; j < topology[i]; ++j) {
            biases[i][j] += learningRate * deltas[i][j];
            for (int k {0}; k < topology[i - 1]; ++k)
                weights[i - 1][j][k] += learningRate * deltas[i][j] * outputs[i - 1][k];
        }
    }
}

void Autoencoder::train(const Matrix& data, int epochs, double threshold) {
    double avgError = std::numeric_limits<double>::max();
    for (int epoch {1}; epoch <= epochs && avgError > threshold; ++epoch) {
        double totalError {0.0};
        for (size_t i {0}; i < data.size(); ++i) {
            const auto& sample = data[i];
            Vector output = feedforward(sample);
            double sampleError = binary_cross_entropy(sample, output);
            totalError += sampleError;
            backpropagate(sample);
            updateWeights();

            if (i % (data.size() / 100) == 0) {
                printProgress((static_cast<double>(i) / data.size()), epoch, sampleError);
            }
        }
        avgError = totalError / data.size();
        printProgress(1.0, epoch, avgError);
        std::cout << std::endl;
    }
}

void Autoencoder::test(const Matrix& data) {
    double totalError {0.0};
    for (size_t i {0}; i < data.size(); ++i) {
        const auto& sample = data[i];
        Vector output = feedforward(sample);
        totalError += binary_cross_entropy(sample, output);
        if (i % (data.size() / 100) == 0) {
            printProgress((static_cast<double>(i) / data.size()), 0, 0.0);
        }
    }
    double avgError = totalError / data.size();
    printProgress(1.0, 0, 0.0);
    std::cout << "\nTeste concluído - Erro médio: " << std::fixed << std::setprecision(4) << avgError << std::endl;
}

void Autoencoder::reconstruct(const Matrix& data) {
    for (int i {0}; i < 10; ++i) {
        auto output = feedforward(data[i]);
        save_pgm(data[i], "input_" + std::to_string(i) + ".pgm");
        save_pgm(output, "reconstructed_" + std::to_string(i) + ".pgm");
        double loss {0};
        for (size_t j {0}; j < output.size(); ++j) {
            loss += std::abs(output[j] - data[i][j]);
        }
        std::cout << "Imagem " << i << " - Loss: " << loss << std::endl;
    }
}

void Autoencoder::save_model(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Erro ao salvar modelo em " + filename);

    size_t layers = topology.size();
    file.write((char*)&layers, sizeof(layers));
    file.write((char*)topology.data(), layers * sizeof(int));

    for (const auto& layer: weights) {
        for (const auto& neuron : layer) {
            file.write((char*)neuron.data(), neuron.size() * sizeof(double));
        }
    }

    for (const auto& layer_biases : biases) {
        file.write((char*)layer_biases.data(), layer_biases.size() * sizeof(double));
    }

    file.close();
}

void Autoencoder::load_model(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Erro ao carregar modelo de " + filename);

    size_t layers;
    file.read((char*)&layers, sizeof(layers));
    std::vector<int> file_topology(layers);
    file.read((char*)file_topology.data(), layers * sizeof(int));

    if (file_topology != topology)
        throw std::runtime_error("Topologia do modelo não corresponde à do autoencoder.");
    
    for (auto& layer : weights) {
        for (auto& neuron : layer) {
            file.read((char*)neuron.data(), neuron.size() * sizeof(double));
        }
    }

    for (auto& layer_biases : biases) {
        file.read((char*)layer_biases.data(), layer_biases.size() * sizeof(double));
    }

    file.close();
}
