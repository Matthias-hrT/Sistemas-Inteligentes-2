#include <iostream>
#include <set>
#include "load_mnist.h"
#include "autoencoder.h"

int main(int argc, char** argv) {
    std::cout << "Carregando Dataset..." << std::endl;
    auto trainImages = load_images("/home/hirata/SI2/datasets/fashion-mnist/train-images.idx3-ubyte", 10000);
    auto testImages = load_images("/home/hirata/SI2/datasets/fashion-mnist/t10k-images.idx3-ubyte", 1000);

    if (argv[1] == std::string("train")) {
        std::cout << "Iniciando Autoencoder..." << std::endl;
        Autoencoder ae({784, 64, 784}, 0.01);
        ae.train(trainImages, 200, 1e-3);
        ae.save_model("autoencoder_model.bin");
    } else if (argv[1] == std::string("test")) {
        std::cout << "Carregando modelo salvo..." << std::endl;
        Autoencoder ae({784, 64, 784}, 0.01);
        ae.load_model("autoencoder_model.bin");
        ae.test(testImages);
        ae.reconstruct(testImages);
    } else {
        std::cerr << "Uso: " << argv[0] << " <train|test>" << std::endl;
        return 1;
    }

    return 0;
}