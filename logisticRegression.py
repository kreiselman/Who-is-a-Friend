import sklearn.svm
import sklearn.metrics
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load data
train = pd.read_csv('train.csv')
X = train.iloc[:,3:5]
y = train['Friends']

Compd = pd.read_csv('test.csv')
CompX = Compd.iloc[:,3:5]

# Split data
xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.5)

# Train Model
logistic = sklearn.linear_model.LogisticRegression()
logistic.fit(xtr, ytr)
yhatSubmit = logistic.predict(CompX)

# Export our answers
ids = list(range(1, yhatSubmit.shape[0] + 1))
df = pd.DataFrame(data={'ID': ids, 'Friends': yhatSubmit})
df.to_csv('answer.csv', index=False)
