import sklearn.svm
import sklearn.metrics
import sklearn.preprocessing
import numpy as np
import pandas
from sklearn.metrics import log_loss

# Load data
d = pandas.read_csv('train.csv')
y = np.array(d.Friends)  # Labels
# X = np.array(d.iloc[:,3:6])  # Features
X = np.array(d.iloc[:,3:5])  # Features

Compd = pandas.read_csv('test.csv')
CompX = np.array(Compd.iloc[:,3:5])  # Features


newList = np.random.permutation(np.arange(X.shape[0]))
randomX = X[newList]
randomY = y[newList]



# for i in range(X.shape[0]):
#     if(randomX[i,2] == "At Work"):
#         randomX[i,2] = .3
#     elif(randomX[i,2] == "Over a Meal"):
#         randomX[i,2] = .9
#     elif (randomX[i,2] == "Social_Media"):
#         randomX[i,2] = .3
#     elif (randomX[i,2] == "Party"):
#         randomX[i,2] = .7
#     elif (randomX[i,2] == "In Passing"):
#         randomX[i,2] = .1
#     elif (randomX[i,2] == "Class"):
#         randomX[i, 2] = .3
#
#
# for i in range(CompX.shape[0]):
#     if(CompX[i,2] == "At Work"):
#         CompX[i,2] = .3
#     elif(CompX[i,2] == "Over a Meal"):
#         CompX[i,2] = .9
#     elif (CompX[i,2] == "Social_Media"):
#         CompX[i,2] = .3
#     elif (CompX[i,2] == "Party"):
#         CompX[i,2] = .7
#     elif (CompX[i,2] == "In Passing"):
#         CompX[i,2] = .1
#     elif (CompX[i,2] == "Class"):
#         CompX[i, 2] = .3


xtr = randomX[0:20000]
ytr = randomY[0:20000]

xte = randomX[20000:40000]
yte = randomY[20000:40000]



# # Linear SVM
# linearSVM = sklearn.svm.LinearSVC(dual = False)
# linearSVM.fit(xtr, ytr)
# print("Lineasr SVM done")
#
# # yhat1 = linearSVM.decision_function(xte)  # Linear kernel
#
# yhatSubmit = linearSVM.decision_function(CompX)  # Linear kernel

# auc1 = sklearn.metrics.roc_auc_score(yte, yhat1)

linear = sklearn.linear_model.LinearRegression()
linear.fit(xtr,ytr)
yhatSubmit = linear.predict(CompX)

yhatSubmit = np.asarray(yhatSubmit)

mean = np.mean(yhatSubmit)
loss = (1/yhatSubmit.shape[0]) * np.sum(yhatSubmit - yte)**2


# prob = logistic.predict_proba(xte)
# loss = log_loss(yte, prob)

print("Loss is ", loss)


response = []

ids = []
friends = []
for i in range(yhatSubmit.shape[0]):
    if (yhatSubmit[i] >= mean):
       ids.append(i + 1)
       friends.append(1)
    else:
       ids.append(i + 1)
       friends.append(0)


print(yhatSubmit)


df = pandas.DataFrame(data={'ID': ids, 'Friends': friends})
df.to_csv('answer-linear.csv', index=False)

print("response is ", response)



# d = {'col1': [1, 2], 'col2': [3, 4]}
# df = pd.DataFrame(data=d)
