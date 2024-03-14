from sklearn.neural_network import MLPClassifier

import pandas as pd

X_credit_train = pd.read_csv("data/trainX.csv")
y_credit_train = pd.read_csv("data/trainY.csv")

rede_neural_credit = MLPClassifier(max_iter=1200, verbose=True, tol=0.0000100,
                                   solver = 'adam', activation = 'relu',
                                   hidden_layer_sizes = (20, 20, 20))
rede_neural_credit.fit(X_credit_train, y_credit_train)

