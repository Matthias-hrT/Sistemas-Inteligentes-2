#include <iostream>     // std::cout, std::endl
#include <vector>       // std::vector
#include <string>       // std::string
#include <fstream>      // std::ifstream, std::ofstream
#include <sstream>      // std::stringstream
#include <cmath>        // std::exp, std::log, std::pow
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>    // std::shuffle, std::max_element    
#include <numeric>      // std::iota
#include <map>          // std::map
#include <utility>      // std::pair
#include <chrono>       // std::chrono::high_resolution_clock::now() std::chrono::chrono::duration_cast<>

#include "gnuplot-iostream.h" // Gnuplot

using namespace std; // Para evitar usar std:: em cada linha

using Matrix = vector<vector<double>>; // Define a Matriz como uma matriz 2D de doubles
using Vector = vector<double>;          // Define o Vetor como um vetor de doubles

/**
 * Função de ativação sigmoid
 * x: valor de entrada
 * retorna: valor da função sigmoid
 */
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/**
 * Derivada da função sigmoid
 * x: valor de entrada (saída da sigmoid)
 * retorna: valor da derivada da função sigmoid
 */
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}


/**
 * Classe MLP (Multi-Layer Perceptron)
 * Implementa uma rede neural feedfoward com:
 * -topology: vetor com o número de neurônios em cada camada 
 * -learningRate: taxa de aprendizado
 * -neuronOutputs: armazena as saídas dos neurônios
 * -deltas: armazena os deltas para backpropagation
 * -weights: armazena os pesos das conexões
 * -biases: armazena os biases dos neurônios
 * -train: método para treinar a rede neural
 * -feedforward: método para realizar a propagação direta
 * -backpropagate: método para realizar a retropropagação
 * -updateWeights: método para atualizar os pesos e biases
 */
class MLP {
    public:
        // Construtor
        MLP(const vector<int>& topology, double learningRate = 0.1);
        
        // Feedforward
        Vector feedforward(const Vector& input);

        // Treinamento com backpropagation
        void train(const Matrix& inputs, const Matrix& targets, int epochs, double threshold, bool plotError = false);

        void saveModel(const string& filename);
        void loadModel(const string& filename);

    private:
        void backpropagate(const Vector& target);
        void updateWeights();

        vector<int> topology;
        double learningRate;
        vector<Vector> neuronOutputs;
        vector<Vector> deltas;
        vector<Matrix> weights;
        vector<Vector> biases;
};

/**
 * Construtor da classe MLP
 * Inicializa a topologia, taxa de aprendizado, pesos e biases
 * topology: vetor com o número de neurônios em cada camada
 * learningRate: taxa de aprendizado
 * retorna: void
 */
MLP::MLP(const vector<int>& topology, double learningRate) 
    : topology(topology), learningRate(learningRate) {
    
    random_device rd;
    //mt19937 gen(rd()); //100% aleatório
    mt19937 g(42);
    uniform_real_distribution<> dis(-1.0, 1.0);
    
    for (size_t i = 0; i < topology.size(); ++i) {
        // Aloca espaço para saídas de deltas de cada camada
        neuronOutputs.push_back(Vector(topology[i]));
        deltas.push_back(Vector(topology[i])); 

        // Aloca bias para cada camada
        biases.push_back(Vector(topology[i]));
        if (i > 0) {
            // Matriz de pesos
            weights.push_back(Matrix(topology[i], Vector(topology[i-1])));
            for (int j = 0; j < topology[i]; ++j) {
                // bias inicial do neurônio j da camada i
                biases[i][j] = dis(g);
                for (int k = 0; k < topology[i-1]; ++k) {
                    // peso entre neurônio k da camada (i-1) e j da camada i
                    weights[i-1][j][k] = dis(g);
                }
            }
        }
    }
}

/**
 * Feedforward
 * input: vetor de entrada
 * retorna: vetor de saída da rede neural
 */
Vector MLP::feedforward(const Vector& input) {
    if (input.size() != topology[0]) {
        throw invalid_argument("Input size does not match network input layer size.");
    }
    
    // Camada de entrada: copia valores de input
    neuronOutputs[0] = input;

    // Para cada camada hidden e de saída
    for (size_t i = 1; i < topology.size(); ++i) {
        for (int j = 0; j < topology[i]; ++j) {
            double sum = 0.0;

            // Soma ponderada das ativações da camada anterior
            for (int k = 0; k < topology[i-1]; ++k) {
                sum += weights[i-1][j][k] * neuronOutputs[i-1][k];
            }

            // Adiciona o bias do neurônio atual
            sum += biases[i][j];

            // Aplica a função de ativação
            neuronOutputs[i][j] = sigmoid(sum);
        }
    }

    // Retorna a saída da última camada
    return neuronOutputs.back();
}

/**
 * Backpropagation
 * target: vetor de saída desejada
 * retorna: void
 */
void MLP::backpropagate(const Vector& target) {
    size_t outputLayerIdx = topology.size() - 1;

    // Camada de saída
    for (int j = 0; j < topology[outputLayerIdx]; ++j) {
        double error = target[j] - neuronOutputs[outputLayerIdx][j];
        deltas[outputLayerIdx][j] = error * sigmoid_derivative(neuronOutputs[outputLayerIdx][j]);
    }

    // Camadas ocultas (de trás para frente, excluindo a camada de entrada)
    for (size_t i = outputLayerIdx - 1; i > 0; --i) {
        for (int j = 0; j < topology[i]; ++j) {
            double errorSum = 0.0;

            // Soma dos erros da camada seguinte ponderados pelos pesos atuais
            for (int k = 0; k < topology[i+1]; ++k) {
                errorSum += weights[i][k][j] * deltas[i+1][k];
            }
               
            // Gradiente local = derivada da ativação * erro propagado
            deltas[i][j] = errorSum * sigmoid_derivative(neuronOutputs[i][j]);
        }
    }
}

/**
 * Atualiza os pesos e biases da rede neural
 * retorna: void
 */
void MLP::updateWeights() {
    for (size_t i = 1; i < topology.size(); ++i) {
        for (int j = 0; j < topology[i]; ++j) {

            // Atualiza bias do neurônio j da camada i
            biases[i][j] += learningRate * deltas[i][j];

            // Atualiza pesos
            for (int k = 0; k < topology[i-1]; ++k) {
                weights[i-1][j][k] += learningRate * deltas[i][j] * neuronOutputs[i-1][k];
            }
        }
    }
}

/**
 * Treina a rede neural usando backpropagation
 * inputs: matriz de entradas
 * targets: matriz de saídas desejadas
 * epochs: número de épocas de treinamento
 * plotError: se true, plota o erro médio por época
 * retorna: void
 */
void MLP::train(const Matrix& inputs, const Matrix& targets, int epochs, double threshold, bool plotError) {
    vector<pair<double, double>> plotData; // Para armazenar dados de plotagem

    double avgError = numeric_limits<double>::max();
    int epoch = 0;

    // Loop de treinamento
    while (epoch < epochs && avgError > threshold) {
        double totalError = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Vector output = feedforward(inputs[i]);
            for (size_t j = 0; j < output.size(); ++j) {
                totalError += 0.5 * pow(targets[i][j] - output[j], 2);
            }
            backpropagate(targets[i]);
            updateWeights();
        }
        avgError = totalError / inputs.size();

        if (plotError) {
            plotData.push_back({(epoch + 1), avgError});
        }

        cout << "Epoch " << (epoch + 1) << " - Avg Error: " << avgError << endl;
        epoch++;
    }

    // Plotagem do erro médio por época
    if (plotError) {
        try {
            cout << "\nGerando grafico" << endl;
            Gnuplot gp;

            gp << "set terminal pngcairo size 800,600\n";
            gp << "set output 'erro_mlp_manual.png'\n";
            gp << "set title 'Convergencia do Erro por Epoca'\n";
            gp << "set xlabel 'Epoca'\n";
            gp << "set ylabel 'Erro Quadratico Medio'\n";
            gp << "set grid\n";
            gp << "set logscale y\n"; // Escala logaritmica no eixo y

            gp << "plot '-' with lines title 'Erro'\n";
            gp.send1d(plotData);
        } catch (const exception &e) {
            cerr << "Erro ao gerar o grafico: " << e.what() << endl;
        }
    }
}

/**
 * Salva o modelo da rede neural em um arquivo
 * filename: nome do arquivo
 * retorna: void
 */
void MLP::saveModel(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Nao foi possivel abrir o arquivo para salvar o modelo: " + filename);
    }

    // Salva a topologia
    for (int neurons : topology) {
        file << neurons << " ";
    }
    file << endl;

    // Salva os pesos
    for (const auto& layerWeights : weights) {
        for (const auto& neuronWeights : layerWeights) {
            for (double weight : neuronWeights) {
                file << weight << " ";
            }
        }
        file << endl;
    }

    // Salva os biases
    for (const auto& layerBiases : biases) {
        for (double bias : layerBiases) {
            file << bias << " ";
        }
        file << endl;
    }

    file.close();
    cout << "Modelo salvo em: " << filename << endl;
}

void MLP::loadModel(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Nao foi possivel abrir o arquivo para carregar o modelo: " + filename);
    }

    // pula a linha da topologia
    string line;
    getline(file, line);

    for (auto& layerWeights : weights) {
        for (auto& neuronWeights : layerWeights) {
            for (double& weight : neuronWeights) {
                file >> weight;
            }
        }
    }

    // Carrega os biases
    for (auto& layerBiases : biases) {
        for (double& bias : layerBiases) {
            file >> bias;
        }
    }

    file.close();
    cout << "Modelo carregado de: " << filename << endl;

}

// Funções para o processamento de dados do Iris =================================

/**
 * Normaliza os dados para o intervalo [0, 1]
 * data: matriz de dados a ser normalizada
 * retorna: void
 */
void normalize(Matrix& data) {

    // Verifica se a matriz não está vazia
    if (data.empty()) return;

    // Obtém o número de colunas
    int numCols = data[0].size();

    // Inicializa os vetores de min e max
    Vector minVals(numCols, numeric_limits<double>::max());
    Vector maxVals(numCols, numeric_limits<double>::lowest());

    // Encontra os valores minimos e maximos para cada coluna
    for (const auto& row : data) {
        for (int i = 0; i < numCols; ++i) {
            if (row[i] < minVals[i]) minVals[i] = row[i];
            if (row[i] > maxVals[i]) maxVals[i] = row[i];
        }
    }

    // Normaliza os dados
    for (auto& row : data) {
        for (int i = 0; i < numCols; ++i) {
            if (maxVals[i] - minVals[i] != 0) {
                row[i] = (row[i] - minVals[i]) / (maxVals[i] - minVals[i]);
            } else {
                row[i] = 0.0; // Evita divisão por zero
            }
        }
    }
}


/**
 * Carrega o dataset do arquivo CSV
 * filename: nome do arquivo CSV
 * inputs: matriz para armazenar as entradas
 * targets: matriz para armazenar as saídas desejadas
 * retorna: void
 */
void load_dataset(const string& filename, Matrix& inputs, Matrix& targets) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Nao foi possivel abrir o arquivo: " + filename);
    }

    // Lê o arquivo linha por linha
    string line;

    getline(file, line); // Ignora o cabeçalho

    // Mapeamento das classes para índices
    map<string, int> labelMap = {
        {"Iris-setosa", 0},
        {"Iris-versicolor", 1},
        {"Iris-virginica", 2}
    };

    // Processa cada linha do arquivo
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        
        getline(ss, cell, ','); // Ignora o ID

        vector<string> values;
        while (getline(ss, cell, ',')) values.push_back(cell);

        if (values.size() < 5) continue; // Verifica se há dados suficientes

        Vector curretFeature;
        for (int i = 0; i < 4; ++i) {
            curretFeature.push_back(stod(values[i]));
        }
        inputs.push_back(curretFeature);

        string class_name = values[4];
        Vector target(3, 0.0);
        if (class_name == "Iris-setosa") target[0] = 1.0;
        else if (class_name == "Iris-versicolor") target[1] = 1.0;
        else if (class_name == "Iris-virginica") target[2] = 1.0;
        targets.push_back(target);
    }
    file.close();
    normalize(inputs);
}

// Lógica para XOR e IRIS =================================================

/**
 * Função para treinar e testar a rede neural no problema XOR
 * retorna: void
 */
void XOR() {
    cout << "Treinando a rede neural para o problema XOR" << endl;

    // Define a topologia da rede neural para o problema XOR
    MLP mlp({2, 2, 1}, 0.5);

    // Dados de entrada e saída para o problema XOR
    Matrix inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    Matrix targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Treinamento
    cout << "Iniciando o treinamento..." << endl;
    mlp.train(inputs, targets, 1000, 1e-1,false);

    // Teste
    cout << "\nResultados apos o treinamento:" << endl;
    for (const auto& input : inputs) {
        Vector output = mlp.feedforward(input);
        cout << "Input: [" << input[0] << ", " << input[1] << "] => Output: " << output[0] << endl;
    }
}

/**
 * Função para treinar e testar a rede neural no dataset Iris
 * retorna: void
 */
void irisTrain() {
    cout << "Treinando a rede neural para o dataset Iris" << endl;

    // Carrega o dataset
    Matrix inputs, targets;
    try {
        load_dataset("Iris.csv", inputs, targets);
    } catch (const exception& e) {
        cerr << e.what() << endl;
        return;
    }

    // Embaralha e divide o dataset em treino (80%) e teste (20%)
    vector<int> indices(inputs.size());
    iota(indices.begin(), indices.end(), 0);
    random_device rd;
    //mt19937 g(rd()); //100% aleatório
    mt19937 g(42);
    shuffle(indices.begin(), indices.end(), g);

    // Cria matrizes embaralhadas
    Matrix shuffledInputs, shuffledTargets;
    for (int idx : indices) {
        shuffledInputs.push_back(inputs[idx]);
        shuffledTargets.push_back(targets[idx]);
    }

    // Divide em treino e teste
    size_t splitIdx = inputs.size() * 0.7;
    Matrix trainInputs(shuffledInputs.begin(), shuffledInputs.begin() + splitIdx);
    Matrix trainTargets(shuffledTargets.begin(), shuffledTargets.begin() + splitIdx);


    // Define a topologia da rede neural para o dataset Iris
    MLP mlp({4, 5, 3}, 0.1);

    // Treinamento
    cout << "Iniciando o treinamento..." << endl;
    auto start = std::chrono::high_resolution_clock::now();
    mlp.train(trainInputs, trainTargets, 5000, 1e-2, true);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout << "Tempo de execução: " << duration.count() << " ms." << endl;

    mlp.saveModel("iris_model.txt");
}

/**
 * Função para testar a rede neural no dataset Iris
 * retorna: void
 */
void irisTest(){
      // Carrega o dataset
    Matrix inputs, targets;
    try {
        load_dataset("Iris.csv", inputs, targets);
    } catch (const exception& e) {
        cerr << e.what() << endl;
        return;
    }

    // Embaralha e divide o dataset em treino (80%) e teste (20%)
    vector<int> indices(inputs.size());
    iota(indices.begin(), indices.end(), 0);
    random_device rd;
    //mt19937 g(rd()); // 100% aleatório
    mt19937 g(42);
    shuffle(indices.begin(), indices.end(), g);

    // Cria matrizes embaralhadas
    Matrix shuffledInputs, shuffledTargets;
    for (int idx : indices) {
        shuffledInputs.push_back(inputs[idx]);
        shuffledTargets.push_back(targets[idx]);
    }

    size_t splitIdx = inputs.size() * 0.3;
    Matrix testInputs(shuffledInputs.begin() + splitIdx, shuffledInputs.end());
    Matrix testTargets(shuffledTargets.begin() + splitIdx, shuffledTargets.end());
    
    cout << "\nResultados no conjunto de teste:" << endl;

    MLP mlpTest({4, 5, 3});
    mlpTest.loadModel("iris_model.txt");

    int acertos = 0;
    for (size_t i = 0; i < testInputs.size(); ++i) {
        Vector output = mlpTest.feedforward(testInputs[i]);
        auto maxIt = max_element(output.begin(), output.end());
        int predicted = distance(output.begin(), maxIt);
        auto targetMaxIt = max_element(testTargets[i].begin(), testTargets[i].end());
        int actual = distance(testTargets[i].begin(), targetMaxIt);
        if (predicted == actual) {
            acertos++;
        }
    }

    // Calcula e exibe a acurácia
    double accuracy = (testInputs.size() > 0) ? (static_cast<double>(acertos) / testInputs.size()) * 100.0 : 0.0;
    cout << "\nAcuracia no conjunto de teste: " << accuracy << "%" << endl;
}

/**
 * Função principal
 * argc: número de argumentos
 * argv: vetor de argumentos
 * retorna: código de saída
 */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " <xor|iris> [train|test]" << endl;
        return 1;
    }

    string mode = argv[1];
    if (mode == "xor") {
        XOR();
    } else if (mode == "iris") {
        if (argc != 3) {
            cerr << "Uso: " << argv[0] << " iris <train|test>" << endl;
            return 1;
        }
        string action = argv[2];
        if (action == "train") {
            irisTrain();
        } else if (action == "test") {
            irisTest();
        } else {
            cerr << "Acao desconhecida: " << action << endl;
            return 1;
        }
    } else {
        cerr << "Modo desconhecido: " << mode << endl;
        return 1;
    }

    return 0;
}