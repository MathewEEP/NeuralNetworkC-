//
// Created by mathew on 2/16/24.
//

#include <cstdio>
#include <cmath>
#include <random>
#include "NeuralNetwork.h"

int main() {
    NeuralNetwork network;
    network.forwardPropagation();
}

NeuralNetwork::NeuralNetwork() {
    this->iteration = 0;
//    this->inputLayer = vector<float>(5);
    this->inputLayer = {0.01, 0.02, 0.03, 0.04, 0.05};
//    this->outputLayer = vector<float>(5);
    this->outputLayer = {2, 3, 5, 4, 1};
    this->hiddenLayer = vector<float>(10);

    this->trainX = { {0.01, 0.02, 0.03, 0.04, 0.05} };
    this->trainY = { {1, 2, 3, 4, 5} };

    // initialize W1, B1, W2, B2 with random numbers from [-0.5, 0.5]
    this->W1 = vector<vector<float>>(hiddenLayer.size(), vector<float>(inputLayer.size(), 1));
    this->B1 = vector<float>(hiddenLayer.size());

    this->W2 = vector<vector<float>>(outputLayer.size(), vector<float>(hiddenLayer.size(), 1));
    this->B2 = vector<float>(outputLayer.size());

    this->initParams();
};


void NeuralNetwork::initParams() {
    const unsigned int seed = time(nullptr);
    mt19937_64 rng(seed);

    uniform_real_distribution<float> unif(-0.5, 0.5);

    for (int i = 0; i < hiddenLayer.size(); i++) {
        this->B1[i] = unif(rng);
        for (int j = 0; j < inputLayer.size(); j++) {
            this->W1[i][j] = unif(rng);
        }
    }

    for (int i = 0; i < outputLayer.size(); i++) {
        this->B2[i] = unif(rng);
        for (int j = 0; j < hiddenLayer.size(); j++) {
            this->W2[i][j] = unif(rng);
        }
    }
}


void NeuralNetwork::dotProduct(vector<float>& inputs, vector<vector<float>>& weights, vector<float>& bias, vector<float>& result) {
    for (int i = 0; i < result.size(); i++) {
        result[i] = bias[i];
        for (int j = 0; j < inputs.size(); j++) {
            result[i] += inputs[j] * weights[i][j];
        }
    }
}

void NeuralNetwork::reLU(vector<float>& inputs) {
    for (float& input : inputs) {
        if (input < 0) {
            input = 0.0f;
        }
    }
}

void NeuralNetwork::sigmoid(vector<float>& inputs) {
    for (float& input : inputs) {
        input = 1 / (1 + exp(-input));
    }
}

void NeuralNetwork::softmax(vector<float>& inputs) {
    float denominator = 0;
    for (float& input: inputs) {
        denominator += exp(input);
    }
    for (float& input: inputs) {
        input = exp(input) / denominator;
    }
}

float NeuralNetwork::crossEntropy() {
    float error = 0;
    for (int i = 0; i < this->outputLayer.size(); i++) {
        error += this->trainY[this->iteration][i] * log(this->outputLayer[i]);
    }
    return -error;
}

float NeuralNetwork::meanSquaredError() {
    float error = 0;
    for (int i = 0; i < this->outputLayer.size(); i++) {
        error += pow((this->trainY[this->iteration][i] - this->outputLayer[i]), 2);
    }
    return error;
}

void NeuralNetwork::calculateOutputDelta(vector<float>& delta) {
    // backpropagation based on https://www.youtube.com/watch?v=sIX_9n-1UbM
    for (int i = 0; i < this->outputLayer.size(); i++) {
        delta[i] = (this->trainY[this->iteration][i] - this->outputLayer[i]) * (this->outputLayer[i]) * (1 - this->outputLayer[i]);
    }
}

void NeuralNetwork::calculateHiddenDelta(vector<float>& delta) {
    
}

void NeuralNetwork::increaseIteration() {
    this->iteration += 1;
}

void NeuralNetwork::forwardPropagation() {
    vector<float> Z1(this->hiddenLayer.size());
    this->dotProduct(this->trainX[this->iteration], this->W1, this->B1, Z1);
    this->sigmoid(Z1);

    vector<float> Z2(this->outputLayer.size());
    this->dotProduct(Z1, this->W2, this->B2, Z2);
}

