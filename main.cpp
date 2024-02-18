//
// Created by mathew on 2/17/24.
//
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>
using namespace std;

const int numInputs = 26;
const int numHiddenNodes = 200;
const int numOutputs = 1;
const int numTrainingSets = 22910;
const int numTestingSets = 5728;

void ReadData(string filenameX, string filenameY, vector<vector<double>>& X, vector<vector<double>>& Y) {
    ifstream src;
    src.open("/home/mathew/CLionProjects/NeuralNetwork/" + filenameX);
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

    src.open("/home/mathew/CLionProjects/NeuralNetwork/" + filenameY);
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

double sigmoid(double x) { return (1 / (1 + exp(-x))); }
double dSigmoid(double x) { return x * (1 - x); }

double init_weights() { return ((double)rand()) / ((double)RAND_MAX); }

int main() {
    const double learningRate = 0.05f;
    vector<double> hiddenLayer(numHiddenNodes);
    vector<double> outputLayer(numOutputs);

    vector<double> hiddenLayerBias(numHiddenNodes);
    vector<double> outputLayerBias(numOutputs);

    vector<vector<double>> hiddenWeights(numInputs, vector<double>(numHiddenNodes));
    vector<vector<double>> outputWeights(numHiddenNodes, vector<double>(numOutputs));

    vector<vector<double>> trainX(numTrainingSets, vector<double>(numInputs));
    vector<vector<double>> trainY(numTrainingSets, vector<double>(numOutputs));
    printf("%zu %zu", trainY.size(), trainY[0].size());

    vector<vector<double>> testX(numTestingSets, vector<double>(numInputs));
    vector<vector<double>> testY(numTestingSets, vector<double>(numOutputs));

    ReadData("trainX.csv", "trainY.csv", trainX, trainY);
    ReadData("testX.csv", "testY.csv", testX, testY);
//
//    ofstream myfile;
//    myfile.open("/home/mathew/CLionProjects/NeuralNetwork/example.txt");
//    ofstream myfile2;
//    myfile2.open("/home/mathew/CLionProjects/NeuralNetwork/example2.txt");
//
//    for (int i = 0; i < trainY.size(); i++) {
//        myfile << trainY[i][0] << " ";
//        for (int j = 0; j < trainX[0].size(); j++) {
//            myfile2 << trainX[i][j] << " ";
//
//        }
//        myfile << "\n";
//        myfile2 << "\n";
//    }
//    myfile.close();
//    myfile2.close();

    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            hiddenWeights[i][j] = init_weights();
        }
    }

    for (int i = 0; i < numHiddenNodes; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i][j] = init_weights();
        }
    }

    for (int j = 0; j < numOutputs; j++) {
        outputLayerBias[j] = init_weights();
    }

    for (int j = 0; j < numOutputs; j++) {
        hiddenLayerBias[j] = init_weights();
    }

    int epochs = 20;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double accuracy = 0;
        for (int x = 0; x < numTrainingSets; x++) {
            // forward prop
            for (int j = 0; j < numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; k++) {
                    activation += trainX[x][k] * hiddenWeights[k][j]; // dot product
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            for (int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j]; // dot product
                }
                outputLayer[j] = sigmoid(activation);
            }

//            printf("Output: %g\tPredicted Output: %g\n", outputLayer[x], trainY[x][0]);
            // back prop
            vector<double> deltaOutput(numOutputs);

            for (int j = 0; j < numOutputs; j++) {
                double error = (trainY[x][j] - outputLayer[j]);
//                printf("ERROR %f\t", error);
                deltaOutput[j] = error; // * dSigmoid(outputLayer[j])
//                printf("ADD %f %f %f\n", trainY[x][j], outputLayer[j], dSigmoid(outputLayer[j]));
            }

            vector<double> deltaHidden(numHiddenNodes);
            for (int j = 0; j < numHiddenNodes; j++) {
                double error = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    error += deltaOutput[k] * outputWeights[j][k];
//                    printf("%f %f\n", deltaOutput[k], outputWeights[j][k]);
                }

                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);

            }

            for (int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * learningRate;
                for (int k = 0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * learningRate;
                }
            }

            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * learningRate;
                for (int k = 0; k < numInputs; k++) {
                    hiddenWeights[k][j] += trainX[x][k] * deltaHidden[j] * learningRate;
//                    printf("ADD %f", deltaHidden[j]);
                }
            }

            // calculate accuracy
            for (int j = 0; j < numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; k++) {
                    activation += trainX[x][k] * hiddenWeights[k][j]; // dot product
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            for (int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j]; // dot product
                }
                outputLayer[j] = sigmoid(activation);
                if (outputLayer[j] > 0.5f) {
                    outputLayer[j] = 1.0f;
                }
                else {
                    outputLayer[j] = 0.0f;
                }
                if (outputLayer[j] == trainY[x][j]) accuracy++;
            }

        }
        printf("Epoch %d\tCorrect %f\tAccuracy %f\n", epoch, accuracy, accuracy/numTrainingSets);
    }

    int correct = 0;
    // testing
//    ofstream myfile;
//    myfile.open("/home/mathew/CLionProjects/NeuralNetwork/example.txt");

    for (int x = 0; x < numTestingSets; x++) {


        // forward prop
        for (int j = 0; j < numHiddenNodes; j++) {
            double activation = hiddenLayerBias[j];
            for (int k = 0; k < numInputs; k++) {
                activation += testX[x][k] * hiddenWeights[k][j]; // dot product
//                myfile << hiddenWeights[k][j] << " ";
            }
//            myfile << "\n";
            hiddenLayer[j] = sigmoid(activation);
        }
//        myfile.close();

        for (int j = 0; j < numOutputs; j++) {
            double activation = outputLayerBias[j];
            for (int k = 0; k < numHiddenNodes; k++) {
                activation += hiddenLayer[k] * outputWeights[k][j]; // dot product
            }
            outputLayer[j] = sigmoid(activation);
//            printf("final activation %f\n", activation);
            if (outputLayer[j] > 0.5f) { // argmax
                outputLayer[j] = 1.0f;
            }
            else {
                outputLayer[j] = 0.0f;
            }
        }

        for (int j = 0; j < numOutputs; j++) {
            if (outputLayer[0] == trainY[x][0]) {
                correct++;
            }
        }
    }
    printf("Correct %d out of %d", correct, numTestingSets);
}

