from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
import random

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

xtr, xte, ytr, yte = train_test_split(X[features], X[target], test_size=0.5, shuffle = False, random_state= False)


#Export to baseline file
nums = []
for x in range(y.shape[0]):
    nums.append(random.randint(0,1))
print(nums)

nums = np.asarray(nums)
pc = np.mean(nums[:27349] == yte)
lossRand = log_loss(yte, nums[:27349])

print("random loss is ", lossRand)
print("random correct is", pc)
idsRand = list(range(1, y.shape[0] + 1))
dfRand = pd.DataFrame(data={'ID': idsRand, 'Friends': nums})
dfRand.to_csv('Baseline.csv', index=False)
