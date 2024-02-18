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
# trainX.to_csv("trainX.csv", index=False)
# trainY.to_csv("trainY.csv", index=False)


df = pd.read_csv("trainX.csv")
print(df.shape)

print(df.head())
print(df.info())
