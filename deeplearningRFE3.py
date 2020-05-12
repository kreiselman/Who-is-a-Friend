import sklearn.metrics
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.feature_selection import RFE
import tensorflow as tf

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

xtr, xte, ytr, yte = train_test_split(X[features], X[target], test_size=0.5, shuffle = False, random_state=False)
xtr, validX, ytr, validY = train_test_split(xtr, ytr, test_size=0.5, shuffle = False, random_state= False)

results = []
epochs = [10,20,30]
batches = [10,20,30]


def Hypereval(model):
    # evaluate the model
    bestAcc = 0
    bestEpoch = 0
    bestBatch = 0
    for batch in batches:
        for epoch in epochs:
            model.fit(xtr, ytr, epochs=epoch, batch_size=batch)
            scores = model.evaluate(validX, validY)
            for i in range(len(scores)):
                print("\n%s: %f%%" % (model.metrics_names[i], scores[i]))
            if scores[1] > bestAcc:
                bestAcc = scores[1]
                bestEpoch = epoch
                bestBatch = batch

    print("Best Acc ", bestAcc)
    print("Best Epoch ", bestEpoch)
    print("Best Batch ", bestBatch)
    return bestEpoch, bestBatch



def findRFE():
    labels = []
    acc = []
    filteredFeat = []

    for i in range(6):
        model = sklearn.linear_model.LogisticRegression()

        rfe = RFE(model, i+1)
        rfe = rfe.fit(xtr, ytr)

        print("\n", "rfe", i+1)
        print(rfe.support_)
        labels.append(rfe.support_)
        print(rfe.score(xte, yte))
        acc.append(rfe.score(xte, yte))

        # prob = rfe.predict_proba(xte)
        # loss1 = log_loss(yte, prob)
        #
        # print("Loss is ", loss1, "\n")

    labels = np.asarray(labels)
    acc = np.asarray(acc)
    bestacc = np.argmax(acc)

    bestLabel = labels[bestacc]

    if bestLabel[0]:
        filteredFeat.append('Person A')
    if bestLabel[1]:
        filteredFeat.append('Person B')
    if bestLabel[2]:
        filteredFeat.append('Years of Knowing')
    if bestLabel[3]:
        filteredFeat.append('Interaction Duration')
    if bestLabel[4]:
        filteredFeat.append('Interaction Type')
    if bestLabel[5]:
        filteredFeat.append('Moon Phase During Interaction')
    return filteredFeat

filtFeat = findRFE()



xtr, xte, ytr, yte = train_test_split(X[filtFeat], X[target], test_size=0.5, shuffle = False, random_state=False)
xtr, validX, ytr, validY = train_test_split(xtr, ytr, test_size=0.5, shuffle = False, random_state= False)

model3 = Sequential()
model3.add(Dense(24, activation='relu', input_dim=len(filtFeat)))
model3.add(Dense(12, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(optimizer='SGD',
 loss='binary_crossentropy',
 metrics=['accuracy'])






bestEpoch, bestBatch = Hypereval(model3)
model3.fit(xte, yte, epochs=bestEpoch, batch_size=bestBatch )
yhatSubmit = np.around(model3.predict(y[filtFeat]))

scores = model3.evaluate(xte, yte)

for i in range(len(scores)):
    print("\nFinal %s: %f%%" % (model3.metrics_names[i], scores[i]))

print(filtFeat)
yhatSubmit = yhatSubmit.astype(int)
print(yhatSubmit)


ids = list(range(1, yhatSubmit.shape[0] + 1))
df = pd.DataFrame(data={'ID': ids, 'Friends': yhatSubmit.flatten()})
df.to_csv('deep_answer3RFE.csv', index=False)
