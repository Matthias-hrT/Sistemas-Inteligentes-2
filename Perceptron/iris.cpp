#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "gnuplot-iostream.h"
#include <random>

using namespace std;

struct Perceptron {
    vector<double> W;
    double bias {1.0};
    double learning_rate {0};
    double eps {1e-12};
    vector<double> avgErrorVector;
    
    Perceptron() {}

    Perceptron(int n_data, double learning_rate, unsigned int seed = 0) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        W = vector<double>(n_data + 1);
        for (auto& weight : W){
            weight = dist(rng);
        }
        this->learning_rate = learning_rate;
    }
    
    double activate (double x) {
        return (x > 0.0) ? 1.0 : 0.0;
    }

    double predict (const vector<double>& inputs) {
        double weighted_sum = bias * W[0];
        for (size_t i = 0; i < inputs.size(); ++i) {
            weighted_sum += W[i + 1] * inputs[i];
        }
        return activate(weighted_sum);
    }

    void train(const vector<vector<double>>& data, int num_inter, bool verbose=true) {
        int epochs {0};
        bool error {true};

        while (error && epochs < num_inter) {
            error = false;
            double avgError {0};

            if (verbose) cout << "\n=== Época " << epochs + 1 << " ===" << endl;

            for (const auto& row: data) {
                double input1 = row[0];
                double input2 = row[1];
                double expected_output = row[2];

                double output = predict({input1, input2});

                avgError += std::pow(std::abs(expected_output - output), 2);

                if (output != expected_output) {
                    double delta = expected_output - output;

                    for (size_t j = 0; j < W.size(); ++j) {
                        W[j] += learning_rate * delta * ((j == 0) ? bias : (j == 1 ? input1 : input2));
                        if (W[j] < eps && W[j] > -eps) {
                            W[j] = 0.0;
                        } 
                    }
                    error = true;
                    cout << "Epoch: " << epochs 
                     << " Adjusting weights: (" 
                     << W[0] << ", " << W[1] << ", " << W[2] << ")" 
                     << " for input: (" << input1 << ", " << input2 << ")" 
                     << " expected: " << expected_output 
                     << " output: " << output << endl;
                }
            }
            avgError /= data.size();
            avgErrorVector.push_back(avgError);
            epochs++;
        }
        if (verbose) cout << "\nTreinamento finalizado em " << epochs << " épocas." << endl;
    }

    double test(const vector<vector<double>>& data) {
        int correct = 0;
        for (const auto& row : data) {
            double output = predict({row[0], row[1]});
            double expected_output = row.back();
            if (output == expected_output) correct++;
        }
        return static_cast<double>(correct) / data.size();
    }

    void save_weights(const string& filename) {
        ofstream file(filename);
        for (double w : W) file << w << " ";
        file << "\n" << bias;
        file.close();
    }

    void load_weights(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) return;
        for (auto& w : W) file >> w;
        file >> bias;
        file.close();
    }

};

void load_csv(const string& filename, vector<vector<double>>& data) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Erro abrindo arquivo: " << filename << endl;
        return;
    }

    string line;
    getline(file, line); 

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;

        // ignora ID
        getline(ss, cell, ',');

        vector<string> values;
        while (getline(ss, cell, ',')) values.push_back(cell);

        if (values.size() < 1) continue;
        string class_name = values.back();
        values.pop_back();

        // pega valores numéricos
        double sepalLength = stod(values[0]);
        double sepalWidth  = stod(values[1]);
        double petalLength = stod(values[2]);
        double petalWidth  = stod(values[3]);

        // soma
        double sepal = sepalLength + sepalWidth;
        double petal = petalLength + petalWidth;

        // binariza classe (setosa = 0, resto = 1)
        int cls = (class_name == "Iris-setosa") ? 0 : 1;

        // monta linha no estilo das portas lógicas
        data.push_back({sepal, petal, (double)cls});
    }
    file.close();
}


void plot_error(const vector<double>& avgErrorVector) {
    Gnuplot gp_erro;
    gp_erro << "set title 'Avg Error per Epoch'\n";
    gp_erro << "set xlabel 'Epochs'\nset ylabel 'Avg Error'\n";
    gp_erro << "set xrange [0:*]\nset yrange [0:*]\n";
    gp_erro << "plot '-' with linespoints lt rgb 'black' title 'Errors'\n";
    gp_erro.send1d(avgErrorVector);
}

void plot_hyperplane(const vector<vector<double>>& train_data, const vector<vector<double>>& test_data, const vector<double>& W, double bias, double accuracy) {
    // Descobre limites
    double x_min = train_data[0][0], x_max = train_data[0][0];
    double y_min = train_data[0][1], y_max = train_data[0][1];

    auto update_limits = [&](const vector<vector<double>>& data) {
        for (const auto& row : data) {
            x_min = min(x_min, row[0]); x_max = max(x_max, row[0]);
            y_min = min(y_min, row[1]); y_max = max(y_max, row[1]);
        }
    };
    update_limits(test_data);

    // Separa pontos
    vector<pair<double,double>> train0, train1, test0, test1;
    for (const auto& row : train_data) {
        if (row[2]==0) train0.push_back({row[0], row[1]});
        else train1.push_back({row[0], row[1]});
    }
    for (const auto& row : test_data) {
        if (row[2]==0) test0.push_back({row[0], row[1]});
        else test1.push_back({row[0], row[1]});
    }

    // Hiperplano
    double w0 = W[0], w1 = W[1], w2 = W[2];
    vector<pair<double,double>> line;
    if (fabs(w2) > 1e-12) {
        double margin = 1.0;
        for (double x = x_min - margin; x <= x_max + margin; x += (x_max - x_min)/100.0) {
            double y = -(w1 / w2) * x - (w0 / w2);
            line.push_back({x,y});
        }
    }

    // Plot
    Gnuplot gp;
    gp << "set title 'Perceptron - Hiperplano (Iris) | Accuracy: " << accuracy*100 << "%'\n";
    gp << "set xlabel 'Input 1'\nset ylabel 'Input 2'\n";
    gp << "set xrange [" << x_min-1 << ":" << x_max+1 << "]\n";
    gp << "set yrange [" << y_min-1 << ":" << y_max+1 << "]\n";
    gp << "plot '-' with points pt 7 ps 1 lc rgb 'blue' title 'Classe 0 Treino',"
          "'-' with points pt 7 ps 1 lc rgb 'orange' title 'Classe 1 Treino',"
          "'-' with points pt 5 ps 1 lc rgb 'black' title 'Classe 0 Teste',"
          "'-' with points pt 5 ps 1 lc rgb 'purple' title 'Classe 1 Teste',"
          "'-' with lines lc rgb 'red' dt 2 lw 2 title 'Hiperplano'\n";

    gp.send1d(train0);
    gp.send1d(train1);
    gp.send1d(test0);
    gp.send1d(test1);
    gp.send1d(line);
}


int main() {
    vector<vector<double>> data;

    load_csv("Iris.csv", data);

    // embaralhar
    unsigned int seed = 42;
    std::mt19937 rng(seed);
    shuffle(data.begin(), data.end(), rng);


    // split 70/30
    int split_index = static_cast<int>(0.7 * data.size());
    vector<vector<double>> train_data(data.begin(), data.begin() + split_index);
    vector<vector<double>> test_data(data.begin() + split_index, data.end());

    int num_iter {100};
    Perceptron perceptron(2, 0.1);
    perceptron.train(train_data, num_iter);
    perceptron.save_weights("weights_iris.txt");


    perceptron.load_weights("weights_iris.txt");
    double accuracy = perceptron.test(test_data);
    cout << "Acurácia no teste: " << accuracy * 100 << "%" << endl;

    plot_error(perceptron.avgErrorVector);
    plot_hyperplane(train_data, test_data, perceptron.W, perceptron.bias, accuracy);



    return 0;
}
