import sklearn.svm
import sklearn.metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
df = pd.concat([train, test])

check = df.head(1)
# Define our features and target
features = list(set(train.columns) - set(['Friends', 'ID']))
target = 'Friends'

# Encode our categorical data
categorical = ['Person A', 'Person B', 'Interaction Type', 'Moon Phase During Interaction']
le = preprocessing.LabelEncoder()

for feature in categorical:
    df[feature] = le.fit_transform(df[feature])

# Split data
X = df[df[target].notnull()]
y = df[df[target].isnull()]
X[target] = X[target].astype(int)

xtr, xte, ytr, yte = train_test_split(X[features], X[target], test_size=0.5, shuffle = False)

# Train Model
logistic = sklearn.linear_model.LogisticRegression()
logistic.fit(xtr, ytr)

prob = logistic.predict_proba(xte)
loss = log_loss(yte, prob)

print("Loss is ", loss)

yhatSubmit = logistic.predict(y[features])

# Feature Importance (still testing to get it right)
importance = logistic.coef_[0]
print("Score of all features: ", logistic.score(xte, yte))

for i, v in enumerate(importance):
	print('Feature:', features[i], 'Importance:', v)

# Export our answers
ids = list(range(1, yhatSubmit.shape[0] + 1))
df = pd.DataFrame(data={'ID': ids, 'Friends': yhatSubmit})
df.to_csv('answer.csv', index=False)

model =  sklearn.linear_model.LogisticRegression()

rfe1 = RFE(model, 1)
rfe1 = rfe1.fit(xtr, ytr)
rfeSubmit1 = rfe1.predict(y[features])

print("\n", "rfe1")
print(rfe1.support_)
print(rfe1.ranking_)
print(rfe1.score(xte, yte))

# Export our answers
ids1 = list(range(1, rfeSubmit1.shape[0] + 1))
df1 = pd.DataFrame(data={'ID': ids, 'Friends': yhatSubmit})
df1.to_csv('answer_RFE1.csv', index=False)

rfe2 = RFE(model, 2)
rfe2 = rfe2.fit(xtr, ytr)
rfeSubmit2 = rfe2.predict(y[features])

print("\n", "rfe2")
print(rfe2.support_)
print(rfe2.ranking_)
print(rfe2.score(xte, yte))

# Export our answers
ids2 = list(range(1, rfeSubmit2.shape[0] + 1))
df2 = pd.DataFrame(data={'ID': ids, 'Friends': yhatSubmit})
df2.to_csv('answer_RFE2.csv', index=False)

rfe3 = RFE(model, 3)
rfe3 = rfe3.fit(xtr, ytr)
rfeSubmit3 = rfe3.predict(y[features])

print("\n", "rfe3")
print(rfe3.support_)
print(rfe3.ranking_)
print(rfe3.score(xte, yte))

# Export our answers
ids3 = list(range(1, rfeSubmit3.shape[0] + 1))
df3 = pd.DataFrame(data={'ID': ids, 'Friends': yhatSubmit})
df3.to_csv('answer_RFE3.csv', index=False)

rfe4 = RFE(model, 4)
rfe4 = rfe4.fit(xtr, ytr)
rfeSubmit4 = rfe4.predict(y[features])

print("\n", "rfe4")
print(rfe4.support_)
print(rfe4.ranking_)
print(rfe4.score(xte, yte))

# Export our answers
ids4 = list(range(1, rfeSubmit4.shape[0] + 1))
df4 = pd.DataFrame(data={'ID': ids, 'Friends': yhatSubmit})
df4.to_csv('answer_RFE4.csv', index=False)

rfe5 = RFE(model, 4)
rfe5 = rfe5.fit(xtr, ytr)
rfeSubmit5 = rfe5.predict(y[features])

print("\n", "rfe5")
print(rfe5.support_)
print(rfe5.ranking_)
print(rfe5.score(xte, yte))

# Export our answers
ids5 = list(range(1, rfeSubmit5.shape[0] + 1))
df5 = pd.DataFrame(data={'ID': ids, 'Friends': yhatSubmit})
df5.to_csv('answer_RFE5.csv', index=False)