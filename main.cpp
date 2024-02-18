//
// Created by mathew on 2/17/24.
// Followed tutorial https://www.youtube.com/watch?v=LA4I3cWkp1E
// http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html

#include <cstdio>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>
#include <random>

using namespace std;

const int numInputs = 26;
const int numHiddenNodes = 100;
const int numOutputs = 1;
const int numTrainingSets = 22910;
const int numTestingSets = 5728;

const double leakyReLUMultiplier = 0.01;

void ReadData(string filenameX, string filenameY, vector<vector<double>>& X, vector<vector<double>>& Y) {
    ifstream src;
    src.open("/home/mathew/CLionProjects/NeuralNetwork/data/" + filenameX);
    string value;

    getline(src, value);
    int idx = 0;
    int row = 0;
    while (getline(src, value)) {
        stringstream input(value);
        while (idx < numInputs) {
            getline(input, value, ',');

            X[row][idx] = stod(value);

            idx += 1;
        }
        idx = 0;
        row += 1;
    }
    src.close();

    src.open("/home/mathew/CLionProjects/NeuralNetwork/data/" + filenameY);
//    printf("IS OPEN %B\n", src.is_open());
    getline(src, value);
    row = 0;
    while (getline(src, value)) {
        Y[row][0] = stod(value);
//        printf("%f ", stod(value));
        row += 1;
    }
    src.close();
}

void ReadBias(string filename, vector<double>& bias) {
    ifstream src;
    src.open("/home/mathew/CLionProjects/NeuralNetwork/models/" + filename);
    string value;
    getline(src, value);
    stringstream input(value);
    int idx = 0;
    while (getline(input, value, ',')) {
        bias[idx] = stod(value);
//        printf("VALUE %f %f %d\n", stod(value), bias[idx], idx);
        idx += 1;
    }
    src.close();
}

void SaveBias(string filename, vector<double>& bias) {
    ofstream src;
    src.open("/home/mathew/CLionProjects/NeuralNetwork/models/" + filename);
    for (int i = 0; i < bias.size(); i++) {
        src << bias[i] << ",";
    }
    src.close();
}

void ReadWeights(string filename, vector<vector<double>>& weights) {
    ifstream src;
    src.open("/home/mathew/CLionProjects/NeuralNetwork/models/" + filename);
    string value;
    int row = 0;
    while (getline(src, value)) {
        stringstream input(value);
        int idx = 0;
        while (getline(input, value, ',')) {
            weights[row][idx] = stod(value);
//        printf("VALUE %f %f %d\n", stod(value), bias[idx], idx);
            idx += 1;
        }
    }
    src.close();
}

void SaveWeights(string filename, vector<vector<double>>& weights) {
    ofstream src;
    src.open("/home/mathew/CLionProjects/NeuralNetwork/models/" + filename);
    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[i].size(); j++) {
            src << weights[i][j] << ",";
        }
        src << "\n";
    }
    src.close();
}

double sigmoid(double x) { return (1 / (1 + exp(-x))); }
double dSigmoid(double x) { return x * (1 - x); }
double leakyReLU(double x) {
    return max(leakyReLUMultiplier * x, x);
}
double dLeakyReLU(double x) {
    if (x > 0.0f) {
        return 1;
    }
    return leakyReLUMultiplier;
}

void train(int epochs, double learningRate, vector<double>& hiddenLayerBias, vector<double>& outputLayerBias, vector<vector<double>>& trainX, vector<vector<double>>& trainY, vector<vector<double>>& hiddenWeights, vector<vector<double>>& outputWeights, vector<double> hiddenLayer, vector<double> outputLayer) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double accuracy = 0;
        for (int x = 0; x < numTrainingSets; x++) {
            // forward prop
            for (int j = 0; j < numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; k++) {
                    activation += trainX[x][k] * hiddenWeights[k][j]; // dot product
                }
                hiddenLayer[j] = leakyReLU(activation);
            }

            for (int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j]; // dot product
                }
                outputLayer[j] = leakyReLU(activation); // was sigmoid
            }

//            printf("Output: %g\tPredicted Output: %g\n", outputLayer[x], trainY[x][0]);
            // back prop
            vector<double> deltaOutput(numOutputs);

            for (int j = 0; j < numOutputs; j++) {
                double error = (trainY[x][j] - outputLayer[j]);
//                printf("ERROR %f\t", error);
                deltaOutput[j] = error * dLeakyReLU(outputLayer[j]); // * dSigmoid(outputLayer[j]) (DEAD NEURON.... https://www.linkedin.com/advice/3/how-do-you-debug-dead-neurons-neural-network-gqxxc)
//                printf("ADD %f %f %f\n", trainY[x][j], outputLayer[j], dSigmoid(outputLayer[j]));
            }

            vector<double> deltaHidden(numHiddenNodes);
            for (int j = 0; j < numHiddenNodes; j++) {
                double error = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    error += deltaOutput[k] * outputWeights[j][k];
//                    printf("%f %f\n", deltaOutput[k], outputWeights[j][k]);
                }

                deltaHidden[j] = error * dLeakyReLU(hiddenLayer[j]);

            }

            for (int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * learningRate;
                for (int k = 0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * learningRate;
//                    printf("%f %f\n", hiddenLayer[k], deltaOutput[j]);
                }
            }

            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * learningRate;
                for (int k = 0; k < numInputs; k++) {
                    hiddenWeights[k][j] += trainX[x][k] * deltaHidden[j] * learningRate;
//                    printf("ADD %f", deltaHidden[j]);
                }
            }

//            // calculate accuracy
//            for (int j = 0; j < numHiddenNodes; j++) {
//                double activation = hiddenLayerBias[j];
//                for (int k = 0; k < numInputs; k++) {
//                    activation += trainX[x][k] * hiddenWeights[k][j]; // dot product
//                }
//                hiddenLayer[j] = leakyReLU(activation);
//            }
//
//            for (int j = 0; j < numOutputs; j++) {
//                double activation = outputLayerBias[j];
//                for (int k = 0; k < numHiddenNodes; k++) {
//                    activation += hiddenLayer[k] * outputWeights[k][j]; // dot product
//                }
//                outputLayer[j] = leakyReLU(activation);
//                if (outputLayer[j] > 0.5f) {
//                    outputLayer[j] = 1.0f;
//                }
//                else {
//                    outputLayer[j] = 0.0f;
//                }
//                if (outputLayer[j] == trainY[x][j]) accuracy++;
//            }

        }
        printf("Epoch %d\n", epoch);
    }
    SaveBias("1H100N200E/hiddenLayerBias.txt", hiddenLayerBias);
    SaveBias("1H100N200E/outputLayerBias.txt", outputLayerBias);

    SaveWeights("1H100N200E/hiddenWeights.txt", hiddenWeights);
    SaveWeights("1H100N200E/outputWeights.txt", outputWeights);
}

void init_parameters(vector<vector<double>>& hiddenWeights, vector<vector<double>>& outputWeights, vector<double>& hiddenLayerBias, vector<double>& outputLayerBias) {
    const unsigned int seed = time(nullptr);
    mt19937_64 rng(seed);

    uniform_real_distribution<float> unif(0.0, 1.0);

    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            hiddenWeights[i][j] = unif(rng);
        }
    }

    for (int i = 0; i < numHiddenNodes; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i][j] = unif(rng);
        }
    }

    for (int j = 0; j < numOutputs; j++) {
        outputLayerBias[j] = unif(rng);
    }

    for (int j = 0; j < numHiddenNodes; j++) {
        hiddenLayerBias[j] = unif(rng);
    }
}

int main() {
    const double learningRate = 0.001f;
    vector<double> hiddenLayer(numHiddenNodes);
    vector<double> outputLayer(numOutputs);

    vector<double> hiddenLayerBias(numHiddenNodes);
    vector<double> outputLayerBias(numOutputs);

    vector<vector<double>> hiddenWeights(numInputs, vector<double>(numHiddenNodes));
    vector<vector<double>> outputWeights(numHiddenNodes, vector<double>(numOutputs));

    vector<vector<double>> trainX(numTrainingSets, vector<double>(numInputs));
    vector<vector<double>> trainY(numTrainingSets, vector<double>(numOutputs));
//    printf("%zu %zu", trainY.size(), trainY[0].size());

    vector<vector<double>> testX(numTestingSets, vector<double>(numInputs));
    vector<vector<double>> testY(numTestingSets, vector<double>(numOutputs));

    ReadData("trainX.csv", "trainY.csv", trainX, trainY);
    ReadData("testX.csv", "testY.csv", testX, testY);

    init_parameters(hiddenWeights, outputWeights, hiddenLayerBias, outputLayerBias);
    train(50, learningRate, hiddenLayerBias, outputLayerBias, trainX, trainY, hiddenWeights, outputWeights, hiddenLayer, outputLayer);
//
//    int correct = 0;
//    // testing
////    ofstream myfile;
////    myfile.open("/home/mathew/CLionProjects/NeuralNetwork/example.txt");
//
//    for (int x = 0; x < numTestingSets; x++) {
//
//
//        // forward prop
//        for (int j = 0; j < numHiddenNodes; j++) {
//            double activation = hiddenLayerBias[j];
//            for (int k = 0; k < numInputs; k++) {
//                activation += testX[x][k] * hiddenWeights[k][j]; // dot product
////                myfile << hiddenWeights[k][j] << " ";
//            }
////            myfile << "\n";
//            hiddenLayer[j] = leakyReLU(activation);
//        }
////        myfile.close();
//
//        for (int j = 0; j < numOutputs; j++) {
//            double activation = outputLayerBias[j];
//            for (int k = 0; k < numHiddenNodes; k++) {
//                activation += hiddenLayer[k] * outputWeights[k][j]; // dot product
//            }
//            outputLayer[j] = leakyReLU(activation);
////            printf("final activation %f\n", activation);
//            if (outputLayer[j] > 0.5f) { // argmax
//                outputLayer[j] = 1.0f;
//            }
//            else {
//                outputLayer[j] = 0.0f;
//            }
//        }
//
//        for (int j = 0; j < numOutputs; j++) {
//            if (outputLayer[0] == trainY[x][0]) {
//                correct++;
//            }
//        }
//    }
//    printf("Correct %d out of %d", correct, numTestingSets);
}

