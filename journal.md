# CS4243 Final - Journal

## 4/27 - Group Meeting

Today we implemented our first iteration of training using a linear regression model. We submitted the answer to Kaggle and got a private score of 0.49518. We are looking to improve that score, and want to identify certain areas where we can improve the accuracy.


## 4/29 -Meeting with TA
[0,0,0,0,0,0]

[1,0,0,0,0,0]

w_1 = 0.5

one-hot vector

0 min -> 120 min

15 17

5 min interval

## 5/6 - Group Meeting
Improved model accuracy by finding a typo in our code

## 5/6 - Meeting with Prof. Whitehill

- Since our feature space is limited, we should explore these additional featuers:
  - Geography of names and if that has effect
    - https://opendata.stackexchange.com/questions/46/multinational-list-of-popular-first-names-and-surnames

  - Try one hot encoding names and potentially choose the top ~10% to avoid overfitting

  - Try to explore the meeting type and moon phase feature

- We should use a logistic regression model instead of linear regression

- Do the deep learning model - the keras default is ok as long as we explore it and play with the depth of it

# 5/7 - Group Meeting
Converted to a logistic regression model, trained using all features, started determining feature importance

# 5/9 Justin
- Tried to understand feature importance. Learned that the negative values
indicate that it is used to determine if the value is 0. The positives have a 
higher impact on resulting in a 1.
- Found out about rfe which calculates the feature importance and recursively 
finds the most important features to use. Added 5 rfe to see if adding features affects
the percentage correct. It seems to depend on the selected data from the randomized training set
but it seems to level off around 3/4 features.

#5/10 Group Work
-Started writing the paper and creating the presentation
-Separated the deep learning models beased on depth to see if there is a correlation between depth and accuracy
-added accuracy to log and linear regression.
-added random baseline to see if guessing results is equivalent to the models we made

