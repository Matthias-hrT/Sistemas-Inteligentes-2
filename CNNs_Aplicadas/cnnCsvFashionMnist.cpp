#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

static void construct_cnn(network<sequential>& nn) {
    nn << convolutional_layer<relu>(32, 32, 5, 1, 6, padding::valid, true, 1, 1)
       << average_pooling_layer(28, 28, 6, 2)

       << convolutional_layer<relu>(14, 14, 5, 6, 16, padding::valid, true, 1, 1)
       << average_pooling_layer(10, 10, 16, 2)

       << convolutional_layer<relu>(5, 5, 5, 16, 120, padding::valid, true, 1, 1)

       << fully_connected_layer<relu>(120, 84, true)
       << fully_connected_layer<softmax>(84, 10, true);
}

static void compute_metrics(network<sequential>& nn,
                            const std::vector<vec_t>& images,
                            const std::vector<label_t>& labels,
                            double& acc, double& loss)
{
    size_t correct = 0;
    double total_loss = 0.0;

    for (size_t i = 0; i < images.size(); ++i) {
        vec_t y = nn.predict(images[i]);
        label_t pred = std::distance(y.begin(), std::max_element(y.begin(), y.end()));
        if (pred == labels[i]) correct++;
        double p = std::max(1e-15, static_cast<double>(y[labels[i]]));
        total_loss += -std::log(p);
    }
    acc = static_cast<double>(correct) / images.size();
    loss = total_loss / images.size();
}

static void train(const std::string& data_dir_path) {
    network<sequential> nn;
    adam optimizer;

    construct_cnn(nn);

    std::cout << "load models..." << std::endl;

    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
    parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

    std::cout << "start training" << std::endl;

    progress_display disp(train_images.size());
    timer t;
    int minibatch_size = 10;
    int num_epochs = 10;

    std::ofstream csv("metrics.csv");
    csv << "epoch,train_loss,train_acc,test_loss,test_acc,learning_rate,elapsed_seconds\n";
    int current_epoch {0};
    auto on_enumerate_epoch = [&]() {
        double train_acc, train_loss;
        double test_acc, test_loss;
        current_epoch++;

        compute_metrics(nn, train_images, train_labels, train_acc, train_loss);
        compute_metrics(nn, test_images, test_labels, test_acc, test_loss);

        double learning_rate = optimizer.alpha;
        std::cout << "\nEpoch done "<< current_epoch <<"| Train acc: " << std::fixed << std::setprecision(4) << train_acc
                  << " | Test acc: " << test_acc
                  << " | Train loss: " << train_loss
                  << " | Test loss: " << test_loss
                  << " | Learning rate: " << learning_rate
                  << " | Time: " << t.elapsed() << "s\n";

        csv << std::fixed << std::setprecision(6)
            << current_epoch << ","
            << train_loss << ","
            << train_acc << ","
            << test_loss << ","
            << test_acc << ","
            << learning_rate << ","
            << t.elapsed() << "\n";
        csv.flush();
        t.restart();
        disp.restart(train_images.size());
    };

    auto on_enumerate_minibatch = [&]() {
        disp += minibatch_size;
    };

    nn.train<mse>(
        optimizer,
        train_images,
        train_labels,
        minibatch_size,
        num_epochs,
        on_enumerate_minibatch,
        on_enumerate_epoch
    );

    csv.close();

    std::cout << "end training." << std::endl;

    nn.test(test_images, test_labels).print_detail(std::cout);

    std::ofstream ofs("LeNet-model-weight");
    ofs << nn;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage : " << argv[0]
                  << " path_to_data (example:../data)" << std::endl;
        return -1;
    }
    train(argv[1]);
    return 0;
}
