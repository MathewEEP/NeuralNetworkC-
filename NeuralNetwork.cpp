//
// Created by mathew on 2/16/24.
//

#include <cmath>
#include <random>
#include "NeuralNetwork.h"
#include <fstream>
#include <iterator>
#include <sstream>

void ReadData(int rows, int columns, vector<vector<float>>& trainX, vector<vector<float>>& trainY, int rowLimit, vector<vector<float>>& testX, vector<vector<float>>& testY) {
    ifstream src;
    src.open("/home/mathew/CLionProjects/NeuralNetwork/trainX.csv");
    string value;
    getline(src, value);
    int idx = 0;
    int row = 0;
    while (getline(src, value)) {
        stringstream input(value);
        while (idx < columns) {
            getline(input, value, ',');

            if (row > rowLimit) { // take last rows as testX
                testX[rows - row][idx] = stof(value);
            }
            else {
                trainX[row][idx] = stof(value);
            }
            idx += 1;
        }
        idx = 0;
        row += 1;
    }
    src.close();

    src.open("/home/mathew/CLionProjects/NeuralNetwork/trainY.csv");
    getline(src, value);
    row = 0;
    while (getline(src, value)) {
        if (row > rowLimit) { // take last rows as testX
            testY[rows - row][0] = stof(value);
        }
        else {
            trainY[row][0] = stof(value);
        }
        trainY[row][0] = stof(value);
        row += 1;
    }
    src.close();

//    ofstream myfile;
//    myfile.open("/home/mathew/CLionProjects/NeuralNetwork/example.txt");
//
//    for (int i = 0; i < 30; i++) {
//        myfile << trainY[i][0] << " ";
////        for (int j = 0; j < columns; j++) {
////            myfile << trainY[i][j] << " ";
////
////        }
//        myfile << "\n";
//    }
//    myfile.close();
}

int main() {

//    network.forwardPropagation();
//    network.outputLayerDelta();
//    network.hiddenLayerWeightsDerivative();
//    network.hiddenLayerDelta();
//    network.gradientDescent(1000);
    int rows = 28638; // there are 28638 rows in the trainX csv file. save last 138 for test
    int columns = 26;

    vector<vector<float>> trainX = vector<vector<float>>(rows, vector<float>(columns, 1));
    vector<vector<float>> trainY = vector<vector<float>>(rows, vector<float>(1, 1));

    int testSize = 138; // saving last 138 for test
    vector<vector<float>> testX = vector<vector<float>>(testSize, vector<float>(columns, 1));
    vector<vector<float>> testY = vector<vector<float>>(testSize, vector<float>(1, 1));
    ReadData(rows, columns, trainX, trainY, rows - testSize, testX, testY);

    NeuralNetwork network(trainX, trainY, 10);
    network.gradientDescent(200, 0.05);
    int correct = 0;
    for (int i = 0; i < testSize; i++) {
        vector<float> pred = network.forwardPropagation(testX[i]);
        if (pred[0] > 0.5) { // arg max
            pred[0] = 1.0f;
        }
        else {
            pred[0] = 0.0f;
        }
        if (pred[0] == testY[i][0]) correct += 1;
    }
    float percentage = (100 * correct) / (correct + testSize);
    printf("CORRECT %d OUT OF %d, ACCURACY %f", correct, testSize, percentage);
}


NeuralNetwork::NeuralNetwork(vector<vector<float>> trainX, vector<vector<float>> trainY, int numberHiddenLayerNodes) {
    this->numberInputLayerNodes = trainX[0].size();
    this->numberOutputLayerNodes = trainY[0].size();
    this->numberHiddenLayerNodes = numberHiddenLayerNodes;

    this->W1 = vector<vector<float>>(numberHiddenLayerNodes, vector<float>(numberInputLayerNodes, 1));
    this->B1 = vector<float>(numberHiddenLayerNodes);
    this->W2 = vector<vector<float>>(numberOutputLayerNodes, vector<float>(numberHiddenLayerNodes, 1));
    this->B2 = vector<float>(numberOutputLayerNodes);

    this->trainX = trainX;
    this->trainY = trainY;

    this->initParams();
};

void NeuralNetwork::initParams() {
    const unsigned int seed = time(nullptr);
    mt19937_64 rng(seed);

    uniform_real_distribution<float> unif(-0.5, 0.5);

    for (int i = 0; i < this->numberHiddenLayerNodes; i++) {
        this->B1[i] = unif(rng);
        for (int j = 0; j < this->numberInputLayerNodes; j++) {
            this->W1[i][j] = unif(rng);
        }
    }

    for (int i = 0; i < this->numberOutputLayerNodes; i++) {
        this->B2[i] = unif(rng);
        for (int j = 0; j < this->numberHiddenLayerNodes; j++) {
            this->W2[i][j] = unif(rng);
        }
    }
}

vector<float> NeuralNetwork::dotProduct(vector<float>& inputs, vector<vector<float>>& weights, vector<float>& bias) {
    vector<float> result(weights.size());
    for (int i = 0; i < result.size(); i++) {
        result[i] = bias[i];
        for (int j = 0; j < inputs.size(); j++) {
            result[i] += inputs[j] * weights[i][j];
        }
    }
    return result;
}

vector<float> NeuralNetwork::sigmoid(vector<float>& inputs) {
    vector<float> result(inputs.size());
    for (int i = 0; i < inputs.size(); i++) {
        result[i] = 1 / (1 + exp(-inputs[i]));
    }
    return result;
}

vector<float> NeuralNetwork::softmax(vector<float>& inputs) {
    vector<float> result(inputs.size());
    float denominator = 0;
    for (float& input: inputs) {
        denominator += exp(input);
    }
    for (int i = 0; i < inputs.size(); i++) {
        result[i] = exp(inputs[i]) / denominator;
    }
    return result;
}

float NeuralNetwork::meanSquaredError() {
    float error = 0;
    for (int i = 0; i < this->numberOutputLayerNodes; i++) {
        error += pow((this->trainY[this->trainingIndex][i] - this->A2[i]), 2);
    }
    return error / this->numberOutputLayerNodes;
}

void NeuralNetwork::nextTrainingIndex() {
    this->trainingIndex += 1;
}

void NeuralNetwork::resetTrainingIndex() {
    this->trainingIndex = 0;
}

void NeuralNetwork::forwardPropagation() {
    this->Z1 = this->dotProduct(this->trainX[this->trainingIndex], this->W1, this->B1);
    this->A1 = this->sigmoid(Z1);

    this->Z2 = this->dotProduct(Z1, this->W2, this->B2);
    this->A2 = this->sigmoid(Z2); // was using softmax, but since there is only one output, we cannot take the error cost
}

vector<float> NeuralNetwork::outputLayerDelta() {
    // gradient of cost with respect to Z2
    // dC/Z2 = dC/A2 * dA2/Z2
    // cost is mean squared error defined as 1/2m summation(y_actual - y_pred)^2
    // https://www.youtube.com/watch?v=-zI1bldB8to

    // THIS IS ALSO dC/b2 since dZ2/b2 = 1 (b2 is constant)
    vector<float> deltas(A2.size());
    for (int i = 0; i < A2.size(); i++) {
        deltas[i] = (A2[i] - this->trainY[trainingIndex][i]) * (A2[i] * (1 - A2[i]));
//        printf("%f\n", deltas[i]);
    }
    return deltas;
}

vector<float> NeuralNetwork::hiddenLayerDelta() {
    // dZ2/Z1 = dZ2/A1 * dA1/Z1

    // this is also dC/b1
    vector<float> deltas(A1.size());
    for (int i = 0; i < A1.size(); i++) {
        deltas[i] = (A1[i]) * (A1[i]) * (1 - A1[i]);
//        printf("%f\n", deltas[i]);
    }
    return deltas;

}

vector<vector<float>> NeuralNetwork::hiddenLayerWeightsDerivative() {
    // dC/W2 = dC/A2 * dA2/Z2 * dZ2/W2

    vector<vector<float>> dW2 = vector<vector<float>>(numberOutputLayerNodes, vector<float>(numberHiddenLayerNodes, 1));;
    vector<float> outputLayerDelta = this->outputLayerDelta();
    for (int i = 0; i < W2.size(); i++) {
        for (int j = 0; j < W2[i].size(); j++) {
            dW2[i][j] = W2[i][j] * outputLayerDelta[i];
//            printf("%f\n", dW2[i][j]);
        }
    }

    return dW2;
}

vector<vector<float>> NeuralNetwork::inputLayerWeightsDerivative() {
    vector<vector<float>> dW1 = vector<vector<float>>(numberHiddenLayerNodes, vector<float>(numberInputLayerNodes, 1));
    vector<float> hiddenLayerDelta = this->hiddenLayerDelta();
    for (int i = 0; i < W1.size(); i++) {
        for (int j = 0; j < W1[i].size(); j++) {
            dW1[i][j] = W1[i][j] * hiddenLayerDelta[i];
//            printf("%f\n", dW1[i][j]);
        }
    }
    return dW1;
}

void NeuralNetwork::updateParameters(float learningRate) {
    vector<vector<float>> inputLayerWeightsDerivative = this->inputLayerWeightsDerivative();
    for (int i = 0; i < W1.size(); i++) {
        for (int j = 0; j < W1[i].size(); j++) {
            W1[i][j] -= learningRate * inputLayerWeightsDerivative[i][j];
        }
    }
    vector<float> hiddenLayerDelta = this->hiddenLayerDelta();
    for (int i = 0; i < B1.size(); i++) {
        B1[i] -= learningRate * hiddenLayerDelta[i];
    }

    vector<vector<float>> hiddenLayerWeightsDerivative = this->hiddenLayerWeightsDerivative();
    for (int i = 0; i < W2.size(); i++) {
        for (int j = 0; j < W2[i].size(); j++) {
            W2[i][j] -= learningRate * hiddenLayerWeightsDerivative[i][j];
        }
    }
    vector<float> outputLayerDelta = this->outputLayerDelta();
    for (int i = 0; i < B2.size(); i++) {
        B2[i] -= learningRate * outputLayerDelta[i];
    }
}

vector<float> NeuralNetwork::forwardPropagation(vector<float>& input) {
    this->Z1 = this->dotProduct(input, this->W1, this->B1);
    this->A1 = this->sigmoid(Z1);

    this->Z2 = this->dotProduct(Z1, this->W2, this->B2);
    this->A2 = this->sigmoid(Z2);
    return this->A2;
}

void NeuralNetwork::gradientDescent(int steps, float learningRate) {
    for (int i = 0; i < steps; i++) {
        for (int j = 0; j < 5; j++) { // this->trainX.size()
            this->forwardPropagation();
            this->updateParameters(learningRate);
            this->nextTrainingIndex();
        }
        this->resetTrainingIndex();
        if (i % 50 == 0) {
            printf("Iteration %d, Error %f\n", i, this->meanSquaredError());
        }
    }
}