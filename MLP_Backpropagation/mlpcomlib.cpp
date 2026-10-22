#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/rand.h>

#include "gnuplot-iostream.h"

using namespace std;
using namespace dlib;

using sample_type = matrix<float, 4, 1>;
using label_type = unsigned long;

// Converte inteiro em vetor one-hot
matrix<float,3,1> one_hot(unsigned long c) {
    matrix<float,3,1> v;
    v = 0; // inicializa todos os elementos para zero
    if (c < 3) v(c) = 1;
    return v;
}

void LoadIris(const string& filename, std::vector<sample_type>& samples, std::vector<label_type>& labels) {
    ifstream file(filename);
    string line;
    getline(file, line); // pula header

    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        string cell;

        getline(ss, cell, ',');

        sample_type sample;
        bool error = false;
        for (int i = 0; i < 4; ++i) {
            if (!getline(ss, cell, ',')) { error = true; break; }
            try {
                sample(i) = stof(cell);
            } catch(...) { error = true; break; }
        }

        if (error || !getline(ss, cell, ',')) continue;

        samples.push_back(sample);

        cell.erase(remove(cell.begin(), cell.end(), '\"'), cell.end());
        cell.erase(remove(cell.begin(), cell.end(), '\r'), cell.end());

        if (cell == "Iris-setosa") labels.push_back(0);
        else if (cell == "Iris-versicolor") labels.push_back(1);
        else labels.push_back(2);
    }
}

int main() {
    dlib::rand rnd;
    rnd.set_seed("42");

    std::vector<sample_type> all_samples;
    std::vector<label_type> all_labels;

    LoadIris("Iris.csv", all_samples, all_labels);

    randomize_samples(all_samples, all_labels);

    vector_normalizer<sample_type> normalizer;
    normalizer.train(all_samples);

    for (auto& sample: all_samples) {
        sample = normalizer(sample);
    }

    size_t split_idx = all_samples.size() * 0.7;
    std::vector<sample_type> training_samples(all_samples.begin(), all_samples.begin() + split_idx);
    std::vector<label_type> training_labels(all_labels.begin(), all_labels.begin() + split_idx);
    std::vector<sample_type> testing_samples(all_samples.begin() + split_idx, all_samples.end());
    std::vector<label_type> testing_labels(all_labels.begin() + split_idx, all_labels.end());

    using net_type = loss_multiclass_log<
        fc<3, sig<fc<5, sig<input<sample_type>>>>>>;

    net_type net;

    //dnn_trainer<net_type, adam> trainer(net, adam(0.0001, 0.09, 0.999));
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.1);
    trainer.set_learning_rate_shrink_factor(1.0);
    trainer.set_mini_batch_size(1);
    trainer.be_verbose();

    double threshold = 1e-2;

    cout << "\nTreino com dlib" << endl;
    std::vector<pair<double, double>> plot_data;

    const int total_training_steps = 5000;

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < total_training_steps; ++step) {
        trainer.train_one_step(training_samples, training_labels);

        double current_loss = trainer.get_average_loss();
        plot_data.push_back({(double)step, current_loss});
        cout << "Epoch #: " << step + 1 << ", Avg. Error: " << current_loss << endl;

        if (current_loss < threshold) {
            break;
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    cout << "Tempo de execução: " << duration.count() << " ms." << endl;

    try {
        Gnuplot gp;
        gp << "set terminal pngcairo size 800,600\n";
        gp << "set output 'erro_dlib.png'\n";
        gp << "set title 'Convergencia do Erro por Epoca'\n";
        gp << "set xlabel 'Epoca'\n";
        gp << "set ylabel 'Erro Quadratico Medio'\n";
        gp << "set grid\n";
        gp << "set logscale y\n";
        gp << "plot '-' with lines title 'Erro'\n";
        gp.send1d(plot_data);
    } catch (const exception& e) {
        cerr << "Erro ao usar o Gnuplot: " << e.what() << endl;
    }

    auto predicted_labels = net(testing_samples);

    int correct_predictions = 0;
    for (size_t i = 0; i < testing_samples.size(); ++i) {
        long predicted_class = predicted_labels[i];
        long true_class = testing_labels[i];
        if (predicted_class == true_class) {
            correct_predictions++;
        }
    }

    double accuracy = (double)correct_predictions / testing_samples.size() * 100.0;
    cout << "Acuracia do teste: " << accuracy << "%" << endl;

    return 0;
}
