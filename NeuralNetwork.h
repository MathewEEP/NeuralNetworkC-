//
// Created by mathew on 2/16/24.
//

#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

using namespace std;
#include <vector>

class NeuralNetwork {
public:
    int numberInputLayerNodes;
    int numberOutputLayerNodes;
    int numberHiddenLayerNodes; // one hidden layer for simple neural network

    vector<vector<float>> trainX;
    vector<vector<float>> trainY;

    vector<vector<float>> W1;
    vector<float> B1;
    vector<vector<float>> W2;
    vector<float> B2;

    vector<float> A1;
    vector<float> Z1;
    vector<float> A2;
    vector<float> Z2;

    int trainingIndex = 0;
public:
    void initParams();
    void nextTrainingIndex();
    void resetTrainingIndex();

public: // activation functions
    vector<float> softmax(vector<float>& inputs);
    vector<float> sigmoid(vector<float> &inputs);

public: // error functions
    float meanSquaredError();

public: // back propagation
    vector<float> outputLayerDelta();

    vector<vector<float>> hiddenLayerWeightsDerivative();

    vector<float> hiddenLayerDelta();

    vector<vector<float>> inputLayerWeightsDerivative();

    void updateParameters(float learningRate);


public: // forward propagation
    vector<float> dotProduct(vector<float>& inputs, vector<vector<float>>& weights, vector<float>& bias);
    void forwardPropagation();

public:
    void gradientDescent(int steps, float learningRate);

    vector<float> forwardPropagation(vector<float> &input); // to test user input

public:
    NeuralNetwork(vector<vector<float>> trainX, vector<vector<float>> trainY, int numberHiddenLayerNodes, int rows, int columns);
};


#endif //NEURALNETWORK_NEURALNETWORK_H
