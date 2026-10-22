#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

static void construct_cnn(network<sequential>& nn) {
    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6)
       << average_pooling_layer(28, 28, 6, 2)
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16)
       << average_pooling_layer(10, 10, 16, 2)
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120)
       << fully_connected_layer<softmax>(120, 10, true);
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
    int num_epochs = 30;

    optimizer.alpha *= std::sqrt(minibatch_size);

    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_dnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){
        disp += minibatch_size;
    };

    nn.train<cross_entropy_multiclass>(optimizer, train_images, train_labels, minibatch_size, num_epochs, on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    nn.test(test_images, test_labels).print_detail(std::cout);

    nn.save("LeNet-model");

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