import pandas as pd

# def encode_and_bind(original_dataframe, feature_to_encode): # one hot encoding
#     dummies = pd.get_dummies(original_dataframe[[feature_to_encode]]).astype(int)
#     res = pd.concat([original_dataframe, dummies], axis=1)
#     res = res.drop([feature_to_encode], axis=1)
#     return(res)
#
# df = pd.read_csv("credit_risk_dataset.csv")
# df.dropna(inplace=True) # drops missing values
# trainX = df.copy()
# trainX = trainX.drop(columns="loan_status", axis=1)
#
# trainX = encode_and_bind(trainX, "person_home_ownership")
# trainX = encode_and_bind(trainX, "loan_intent")
# trainX = encode_and_bind(trainX, "loan_grade")
# trainX = encode_and_bind(trainX, "cb_person_default_on_file")
#
# for column in trainX.columns: # min-max scaling from https://www.geeksforgeeks.org/data-normalization-with-pandas/
#     trainX[column] = (trainX[column] - trainX[column].min()) / (trainX[column].max() - trainX[column].min())
# print(trainX.shape)
#
# print(trainX.head())
# print(trainX.info())
#
# trainY = df.copy()
# trainY = trainY["loan_status"]
#
# from sklearn.model_selection import train_test_split
#
# trainX, testX = train_test_split(pd.read_csv("trainX.csv"), test_size=0.2)
# trainY, testY = train_test_split(pd.read_csv("trainY.csv"), test_size=0.2)
# testX.to_csv("testX.csv", index=False)
# testY.to_csv("testY.csv", index=False)
# trainX.to_csv("trainX.csv", index=False)
# trainY.to_csv("trainY.csv", index=False)
#
# testX, testY, trainX, trainY = pd.read_csv("testX.csv"), pd.read_csv("testY.csv"), pd.read_csv("trainX.csv"), pd.read_csv("trainY.csv")
# print(testX.shape, testY.shape, trainX.shape, trainY.shape)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

testX, testY, trainX, trainY = pd.read_csv("data/testX.csv"), pd.read_csv("data/testY.csv"), pd.read_csv(
    "data/trainX.csv"), pd.read_csv("data/trainY.csv")

clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=50,activation = 'relu',solver='adam',random_state=1)
clf.fit(trainX, trainY.values.ravel())

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

y_pred = clf.predict(testX)
cm = confusion_matrix(y_pred, testY)
print("Accuracy of MLPClassifier : ''", accuracy(cm))