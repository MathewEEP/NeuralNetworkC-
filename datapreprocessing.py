import pandas as pd

def encode_and_bind(original_dataframe, feature_to_encode): # one hot encoding
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]]).astype(int)
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)

df = pd.read_csv("data/credit_risk_dataset.csv")
df.dropna(inplace=True) # drops missing values
trainX = df.copy()
trainX = trainX.drop(columns="loan_status", axis=1)

trainX = encode_and_bind(trainX, "person_home_ownership")
trainX = encode_and_bind(trainX, "loan_intent")
trainX = encode_and_bind(trainX, "loan_grade")
trainX = encode_and_bind(trainX, "cb_person_default_on_file")

for ind in trainX.index:
    if trainX["person_age"][ind] > 100 or trainX["person_emp_length"][ind] > 100: # weird problem in dataset, including false ages
        print("drop", trainX.loc[ind])
        trainX = trainX.drop(ind)
print(trainX)
trainX.min().to_csv("data/minInputValues.csv", index=False)
trainX.max().to_csv("data/maxInputValues.csv", index=False)
from sklearn.model_selection import train_test_split

trainY = df.copy()
trainY = trainY["loan_status"]

trainX, testX = train_test_split(trainX, test_size=0.2)
trainY, testY = train_test_split(trainY, test_size=0.2)

testX.to_csv("data/testX_unscaled.csv", index=False)
trainX.to_csv("data/trainX_unscaled.csv", index=False)


for column in trainX.columns: # min-max scaling from https://www.geeksforgeeks.org/data-normalization-with-pandas/
    trainX[column] = (trainX[column] - trainX[column].min()) / (trainX[column].max() - trainX[column].min())
    testX[column] = (testX[column] - testX[column].min()) / (testX[column].max() - testX[column].min())



testX.to_csv("data/testX.csv", index=False)
testY.to_csv("data/testY.csv", index=False)
trainX.to_csv("data/trainX.csv", index=False)
trainY.to_csv("data/trainY.csv", index=False)
#
# str = "0.1370967741935483,0.0083388925950633,0.016260162601626,0.144927536231884,0.4387640449438203,0.1204819277108433,0.3571428571428571,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0"
# list = list(map(float, str.split(',')))
# min_ = pd.read_csv("data/minInputValues.csv")
# max_ = pd.read_csv("data/maxInputValues.csv")
# a = []
# for i in range(len(list)):
#     a.append(list[i] * (max_.loc[i]["column"] - min_.loc[i]["column"]) + min_.loc[i]["column"])
# print(a)



# testX, testY, trainX, trainY = pd.read_csv("data/testX.csv"), pd.read_csv("data/testY.csv"), pd.read_csv("data/trainX.csv"), pd.read_csv("data/trainY.csv")
# print(testX.shape, testY.shape, trainX.shape, trainY.shape)
#
# string = "person_age,person_income,person_emp_length,loan_amnt,loan_int_rate,loan_percent_income,cb_person_cred_hist_length,person_home_ownership_MORTGAGE,person_home_ownership_OTHER,person_home_ownership_OWN,person_home_ownership_RENT,loan_intent_DEBTCONSOLIDATION,loan_intent_EDUCATION,loan_intent_HOMEIMPROVEMENT,loan_intent_MEDICAL,loan_intent_PERSONAL,loan_intent_VENTURE,loan_grade_A,loan_grade_B,loan_grade_C,loan_grade_D,loan_grade_E,loan_grade_F,loan_grade_G,cb_person_default_on_file_N,cb_person_default_on_file_Y"
# print(string.split(','))

# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
#
# testX, testY, trainX, trainY = pd.read_csv("data/testX.csv"), pd.read_csv("data/testY.csv"), pd.read_csv(
#     "data/trainX.csv"), pd.read_csv("data/trainY.csv")
#
# clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=50,activation = 'relu',solver='adam',random_state=1)
# clf.fit(trainX, trainY.values.ravel())
#
# def accuracy(confusion_matrix):
#     diagonal_sum = confusion_matrix.trace()
#     sum_of_all_elements = confusion_matrix.sum()
#     return diagonal_sum / sum_of_all_elements
#
# y_pred = clf.predict(testX)
# cm = confusion_matrix(y_pred, testY)
# print("Accuracy of MLPClassifier : ''", accuracy(cm))