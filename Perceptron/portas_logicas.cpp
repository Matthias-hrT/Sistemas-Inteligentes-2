#include <iostream>
#include <vector>
#include <cmath>
#include "gnuplot-iostream.h"

using std::vector;

using namespace std;

int main() {

    // AND
    vector<vector<int>> porta_and {
        {2, 2, 1},
        {4, 4, 0}
    };
    
    // OR
    vector<vector<int>> porta_or {
        {0, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 1}
    };
    

    // XOR
    vector<vector<int>> porta_xor {
        {0, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0}
    };

    // vector<double> W {
    //     {-0.5441, 0.5562, 0.4074}
    // };
    // double bias {-1.0};
    
    vector<double> W {
        {0.5, 0.5, 0.5}
    };
    double bias {1.0};

    double learning_rate {0.1};
    
    int epochs {0};
    int num_inter {100};
    int epoch_errors {0};
    bool error {true};

    vector<double> avgErrorVector;
    vector<vector<int>> X = porta_or;

    vector<pair<int, int>> class0, class1;
    for (auto &ex : X) {
        if (ex[2] == 0) class0.push_back({ex[0], ex[1]});
        else class1.push_back({ex[0], ex[1]});
    }
    
    while (error && epochs < num_inter) {
        error = false;
        double avgError {0};

        for (const auto& example : X) {

            double input1 = example[0];
            double input2 = example[1];
            double expected_output = example[2];

            double weighted_sum = W[0] * bias + W[1] * input1 + W[2] * input2;

            double eps {1e-11};
            double output = (weighted_sum > eps) ? 1.0 : 0.0;

            avgError += std::pow(std::abs(expected_output - output), 2);

            if (output != expected_output) {
                epoch_errors++;
                for (size_t i = 0; i < W.size(); ++i) {
                    W[i] += learning_rate * (expected_output - output) * ((i == 0) ? bias : (i == 1 ? input1 : input2));
                    if (W[i] < eps && W[i] > -eps) {
                        W[i] = 0.0;
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
        avgError /= X.size();
        avgErrorVector.push_back(avgError);
        epochs++;
    }
    cout << "Epoch: " << epochs-1 << ", Weights: (" << W[0] << ", " << W[1] << ", " << W[2] << ")" 
         << endl;

    
    // Plotting
    Gnuplot gp_plano, gp_erro;

    int x_min = X[0][0], x_max = X[0][0];
    int y_min = X[0][1], y_max = X[0][1];
    for (const auto& ex : X) {
        x_min = min(x_min, ex[0]); x_max = max(x_max, ex[0]);
        y_min = min(y_min, ex[1]); y_max = max(y_max, ex[1]);       
    }

    vector<pair<double,double>> line;
    double margin = 1.0;
    if (fabs(W[2]) > 1e-12) {
        for (double x = x_min - margin; x <= x_max + margin; x += (x_max - x_min)/100.0) {
            double y = -(W[1] / W[2]) * x - (W[0] / W[2]);  // equação da linha
            line.push_back({x, y});
        }
    }

    gp_plano << "set title 'Perceptron Learning Process'\n";
    gp_plano << "set xlabel 'X1'\nset ylabel 'X2'\n";
    gp_plano << "set xrange [" << x_min-1 << ":" << x_max+1 << "]\nset yrange [" << y_min-1 << ":" << y_max+1 << "]\n";
    gp_plano << "plot '-' with points pt 7 ps 1 lc rgb 'blue' title 'Class 0',"
            "'-' with points pt 5 ps 1 lc rgb 'orange' title 'Class 1',"
            "'-' with lines lc rgb 'red' dt 2 title 'Decision Boundary'\n";
    gp_plano.send1d(class0);
    gp_plano.send1d(class1);
    gp_plano.send1d(line);

    gp_erro << "set title 'Avg Error'\n";
    gp_erro << "set xlabel 'Epochs'\nset ylabel 'Number of Errors'\n";
    gp_erro << "set xrange [0:*]\nset yrange [0:*]\n";
    gp_erro << "plot '-' with linespoints lt rgb 'black' title 'Errors'\n";
    gp_erro.send1d(avgErrorVector);


    return 0;
}