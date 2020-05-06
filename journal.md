# CS4243 Final - Journal

4/27 - Today we implemented our first iteration of training using a linear regression model. We submitted the answer to Kaggle and got a private score of 0.49518. We are looking to improve that score, and want to identify certain areas where we can improve the accuracy.


4/29 -Meeting with TA comments from meeting
[0,0,0,0,0,0]
[1,0,0,0,0,0]
w_1 = 0.5
one-hot vector
0 min -> 120 min
15 17 
5 min interval

5/6 - Meeting with Prof. Whitehill

- Since our feature space is limited, we should explore these additional featuers:
  - Geography of names and if that has effect
    - https://opendata.stackexchange.com/questions/46/multinational-list-of-popular-first-names-and-surnames

  - Try one hot encoding names and potentially choose the top ~10% to avoid overfitting

- We should use a logistic regression model instead of linear

- Do the deep learning model - the keras default is ok as long as we explore it and play with the depth of it
