// Followed tutorial https://www.youtube.com/watch?v=LA4I3cWkp1E
// This is helpful http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html

#include <cstdio>
#include <iostream>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>
#include <random>

using namespace std;

const int numInputs = 14;
const int numHiddenNodes = 10;
const int numOutputs = 1;
const int numTrainingSets = 9924;
const int numTestingSets = 2482;

const double leakyReLUMultiplier = 0.01;

void ReadData(const string& filenameX, const string& filenameY, vector<vector<double>>& X, vector<vector<double>>& Y) {
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

    getline(src, value);
    row = 0;
    while (getline(src, value)) {
        Y[row][0] = stod(value);
        row += 1;
    }
    src.close();
}

void ReadBias(const string& filename, vector<double>& bias) {
    ifstream src;
    src.open("/home/mathew/CLionProjects/NeuralNetwork/models/" + filename);
    string value;
    getline(src, value);
    stringstream input(value);
    int idx = 0;
    while (getline(input, value, ',')) {
        bias[idx] = stod(value);
        idx += 1;
    }
    src.close();
}

void SaveBias(const string& filename, vector<double>& bias) {
    ofstream src;
    src.open("/home/mathew/CLionProjects/NeuralNetwork/models/" + filename);
    for (double b : bias) {
        src << b << ",";
    }
    src.close();
}

void ReadWeights(const string& filename, vector<vector<double>>& weights) {
    ifstream src;
    src.open("/home/mathew/CLionProjects/NeuralNetwork/models/" + filename);
    string value;
    int row = 0;
    while (getline(src, value)) {
        stringstream input(value);
        int idx = 0;
        while (getline(input, value, ',')) {
            weights[row][idx] = stod(value);
            idx += 1;
        }
    }
    src.close();
}

void SaveWeights(const string& filename, vector<vector<double>>& weights) {
    ofstream src;
    src.open("/home/mathew/CLionProjects/NeuralNetwork/models/" + filename);
    for (auto & weight : weights) {
        for (double j : weight) {
            src << j << ",";
        }
        src << "\n";
    }
    src.close();
}

// activation functions and their derivatives
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
double ReLU(double x) {
    return max(x, 0.0);
}
double dReLU(double x) {
    return x > 0.0;
}

double tanH(double x) {
    return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}
double dTanH(double x) {
    return (1 - pow(x, 2));
}


vector<double> forward(vector<double>& input, vector<double>& hiddenLayerBias, vector<double>& outputLayerBias, vector<vector<double>>& hiddenWeights, vector<vector<double>>& outputWeights, vector<double> hiddenLayer, vector<double> outputLayer) {
    // forward propagation for one observation
    for (int j = 0; j < numHiddenNodes; j++) {
        double activation = hiddenLayerBias[j];
        for (int k = 0; k < numInputs; k++) {
            activation += input[k] * hiddenWeights[k][j]; // dot product
        }
        hiddenLayer[j] = leakyReLU(activation);
    }

    for (int j = 0; j < numOutputs; j++) {
        double activation = outputLayerBias[j];
        for (int k = 0; k < numHiddenNodes; k++) {
            activation += hiddenLayer[k] * outputWeights[k][j]; // dot product
        }
        outputLayer[j] = tanH(activation);
    }

    return outputLayer;
}

void train(int epochs, double learningRate, vector<double>& hiddenLayerBias, vector<double>& outputLayerBias, vector<vector<double>>& trainX, vector<vector<double>>& trainY, vector<vector<double>>& hiddenWeights, vector<vector<double>>& outputWeights, vector<double> hiddenLayer, vector<double> outputLayer) {
    // function to train the NN
    for (int epoch = 0; epoch < epochs; epoch++) {
        double accuracy = 0;
        double avgError = 0;
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
                outputLayer[j] = tanh(activation);

                if ((outputLayer[j] > 0.5f && trainY[x][j] == 1) || (outputLayer[j] < 0.5f && trainY[x][j] == 0)) { // calculate accuracy using argmax
                    accuracy++;
                }
                avgError += pow(outputLayer[j] - trainY[x][j], 2.0); // calculate error
            }

            vector<double> deltaOutput(numOutputs);

            for (int j = 0; j < numOutputs; j++) {
                double error = (trainY[x][j] - outputLayer[j]);
                deltaOutput[j] = error; // problems with a dead neuron https://www.linkedin.com/advice/3/how-do-you-debug-dead-neurons-neural-network-gqxxc
            }

            vector<double> deltaHidden(numHiddenNodes);
            for (int j = 0; j < numHiddenNodes; j++) {
                double error = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error; //  * dLeakyReLU(hiddenLayer[j]) (took out derivative of activation function to help the model converge faster)
            }

            for (int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * learningRate;
                for (int k = 0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += outputWeights[k][j] * deltaOutput[j] * learningRate;
                }
            }

            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * learningRate;
                for (int k = 0; k < numInputs; k++) {
                    hiddenWeights[k][j] += trainX[x][k] * deltaHidden[j] * learningRate;
                }
            }
        }
        printf("Epoch %d\tAccuracy %f\tAverage Error %f\n", epoch, accuracy/numTrainingSets, avgError/numTrainingSets);
    }
    SaveBias("1H100N200E/hiddenLayerBias.txt", hiddenLayerBias); // naming convention for folder is # hidden layers, # neurons in hidden layers, # epochs
    SaveBias("1H100N200E/outputLayerBias.txt", outputLayerBias);

    SaveWeights("1H100N200E/hiddenWeights.txt", hiddenWeights);
    SaveWeights("1H100N200E/outputWeights.txt", outputWeights);
}

void importModel(const string& folder, vector<vector<double>>& hiddenWeights, vector<vector<double>>& outputWeights, vector<double>& hiddenLayerBias, vector<double>& outputLayerBias) {
    ReadWeights(folder + "/hiddenWeights.txt", hiddenWeights);
    ReadWeights(folder + "/outputWeights.txt", outputWeights);
    ReadBias(folder + "/hiddenLayerBias.txt", hiddenLayerBias);
    ReadBias(folder + "/outputLayerBias.txt", outputLayerBias);
}

void init_parameters(vector<vector<double>>& hiddenWeights, vector<vector<double>>& outputWeights, vector<double>& hiddenLayerBias, vector<double>& outputLayerBias) {
    const unsigned int seed = time(nullptr);
    mt19937_64 rng(seed);

    uniform_real_distribution<float> bias(0.0, 1.0);
    normal_distribution<double> hidden(0.0, sqrt(2.0/numInputs)); // saw this formula recommended online to randomly initialize weights
    normal_distribution<double> output(0.0, sqrt(2.0/numHiddenNodes));
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            hiddenWeights[i][j] = hidden(rng);
        }
    }

    for (int i = 0; i < numHiddenNodes; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i][j] = output(rng);
        }
    }

    for (int j = 0; j < numOutputs; j++) {
        outputLayerBias[j] = bias(rng);
    }

    for (int j = 0; j < numHiddenNodes; j++) {
        hiddenLayerBias[j] = bias(rng);
    }
}

void importMinMaxInputValues(const string& filename, vector<double>& inputValues) {
    ifstream src;
    src.open("/home/mathew/CLionProjects/NeuralNetwork/data/" + filename);
    string value;

    getline(src, value);
    int row = 0;
    while (getline(src, value)) {
        inputValues[row] = stod(value);
        row += 1;
    }
    src.close();
}

void test(vector<double>& hiddenLayerBias, vector<double>& outputLayerBias, vector<vector<double>>& testX, vector<vector<double>>& testY, vector<vector<double>>& hiddenWeights, vector<vector<double>>& outputWeights, vector<double> hiddenLayer, vector<double> outputLayer) {
    // this function will test the neural network
    int correct = 0;
    double avgError = 0;

    for (int x = 0; x < numTestingSets; x++) {
        // forward prop
        for (int j = 0; j < numHiddenNodes; j++) {
            double activation = hiddenLayerBias[j];
            for (int k = 0; k < numInputs; k++) {
                activation += testX[x][k] * hiddenWeights[k][j]; // dot product
            }
            hiddenLayer[j] = leakyReLU(activation);
        }

        for (int j = 0; j < numOutputs; j++) {
            double activation = outputLayerBias[j];
            for (int k = 0; k < numHiddenNodes; k++) {
                activation += hiddenLayer[k] * outputWeights[k][j]; // dot product
            }
            outputLayer[j] = tanh(activation);
            if (outputLayer[j] > 0.5f) { // argmax
                outputLayer[j] = 1.0f;
            }
            else {
                outputLayer[j] = 0.0f;
            }
        }

        for (int j = 0; j < numOutputs; j++) {
            if (outputLayer[0] == testY[x][0]) {
                correct++;
            }
            avgError += pow(outputLayer[0] - testY[x][0], 2.0);
        }
    }
    printf("Accuracy %d\tAvg error %f", correct/numTestingSets, avgError/numTestingSets);
}

void liveTest(vector<double>& minInputValues, vector<double>& maxInputValues, vector<double>& hiddenLayerBias, vector<double>& outputLayerBias, vector<double>& input, vector<double>& output, vector<vector<double>>& hiddenWeights, vector<vector<double>>& outputWeights, vector<double>& hiddenLayer, vector<double>& outputLayer) {
    // this function takes in live inputs using a Scanner
    vector<string> columns = {"person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "person_home_ownership_MORTGAGE", "person_home_ownership_OTHER", "person_home_ownership_OWN", "person_home_ownership_RENT", "loan_intent_DEBTCONSOLIDATION", "loan_intent_EDUCATION", "loan_intent_HOMEIMPROVEMENT", "loan_intent_MEDICAL", "loan_intent_PERSONAL", "loan_intent_VENTURE", "loan_grade_A", "loan_grade_B", "loan_grade_C", "loan_grade_D", "loan_grade_E", "loan_grade_F", "loan_grade_G", "cb_person_default_on_file_N", "cb_person_default_on_file_Y"};
    cout << "Taking inputs now\n";
    while (true) {
        for (int i = 0; i < numInputs; i++) {
            cout << columns[i] << ": ";
            cin >> input[i];
            input[i] = (input[i] - minInputValues[i]) / (maxInputValues[i] - minInputValues[i]); // min-max scaling
            cout << "Scaled input " << input[i] << "\n";
        }
        output = forward(input, hiddenLayerBias, outputLayerBias, hiddenWeights, outputWeights, hiddenLayer, outputLayer);
        for (int i = 0; i < numOutputs; i++) {
            cout << "Predicted output: " << output[i] << "\n";
        }
    }
}

int main() {
    const double learningRate = 0.01; // alpha
    vector<double> hiddenLayer(numHiddenNodes);
    vector<double> outputLayer(numOutputs);

    vector<double> hiddenLayerBias(numHiddenNodes);
    vector<double> outputLayerBias(numOutputs);

    vector<vector<double>> hiddenWeights(numInputs, vector<double>(numHiddenNodes));
    vector<vector<double>> outputWeights(numHiddenNodes, vector<double>(numOutputs));

    vector<vector<double>> trainX(numTrainingSets, vector<double>(numInputs));
    vector<vector<double>> trainY(numTrainingSets, vector<double>(numOutputs));

    vector<vector<double>> testX(numTestingSets, vector<double>(numInputs));
    vector<vector<double>> testY(numTestingSets, vector<double>(numOutputs));

    ReadData("trainX.csv", "trainY.csv", trainX, trainY);
    ReadData("testX.csv", "testY.csv", testX, testY);

    init_parameters(hiddenWeights, outputWeights, hiddenLayerBias, outputLayerBias);
    train(80, learningRate, hiddenLayerBias, outputLayerBias, trainX, trainY, hiddenWeights, outputWeights, hiddenLayer, outputLayer);
//    importModel("1H100N200E", hiddenWeights, outputWeights, hiddenLayerBias, outputLayerBias);
    test(hiddenLayerBias, outputLayerBias, testX, testY, hiddenWeights, outputWeights, hiddenLayer, outputLayer);

    vector<double> input(numInputs);
    vector<double> output(numOutputs);

    vector<double> minInputValues(numInputs); // min/max values for scaling purposes of the live test
    vector<double> maxInputValues(numInputs); // this is because the data was scaled using min/max for training the NN

    importMinMaxInputValues("minInputValues.csv", minInputValues);
    importMinMaxInputValues("maxInputValues.csv", maxInputValues);
    liveTest(minInputValues, maxInputValues, hiddenLayerBias, outputLayerBias, input, output, hiddenWeights, outputWeights, hiddenLayer, outputLayer) {
}

