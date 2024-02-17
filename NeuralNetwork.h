//
// Created by mathew on 2/16/24.
//

#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

using namespace std;
#include <vector>

class NeuralNetwork {
public:
    vector<float> inputLayer;
    vector<float> outputLayer;
    vector<float> hiddenLayer; // one hidden layer for simple neural network

    vector<vector<float>> trainX;
    vector<vector<float>> trainY;

    vector<vector<float>> W1;
    vector<float> B1;
    vector<vector<float>> W2;
    vector<float> B2;

    int iteration;
public:
    void initParams();
    void increaseIteration();

public: // activation functions
    void reLU(vector<float>& inputs);
    void softmax(vector<float>& inputs);
    void sigmoid(vector<float> &inputs);

public: // error functions
    float crossEntropy();
    float meanSquaredError();

public: // back propagation
    void calculateOutputDelta(vector<float>& delta);
    void calculateHiddenDelta(vector<float>& delta);

public: // forward propagation
    void dotProduct(vector<float>& inputs, vector<vector<float>>& weights, vector<float>& bias, vector<float>& result);
    void forwardPropagation();

public:
    NeuralNetwork();
};


#endif //NEURALNETWORK_NEURALNETWORK_H
