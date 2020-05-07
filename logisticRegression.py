import sklearn.svm
import sklearn.metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
df = pd.concat([train, test])

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

xtr, xte, ytr, yte = train_test_split(X[features], X[target], test_size=0.5)

# Train Model
logistic = sklearn.linear_model.LogisticRegression()
logistic.fit(xtr, ytr)
yhatSubmit = logistic.predict(y[features])

# Feature Importance (still testing to get it right)
importance = logistic.coef_[0]

for i, v in enumerate(importance):
	print('Feature:', features[i], 'Importance:', v)

# Export our answers
ids = list(range(1, yhatSubmit.shape[0] + 1))
df = pd.DataFrame(data={'ID': ids, 'Friends': yhatSubmit})
df.to_csv('answer.csv', index=False)
