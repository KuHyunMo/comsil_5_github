import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("./train.csv")
test = pd.read_csv("test.csv")
print(train.shape, test.shape)
for col in train.columns:
  colname = col
  colnum = train[col].isna().sum()/train.shape[0]*100
  print("%-12s  -->  %-4.2f%%" %(colname, colnum))

# 결측치가 하나 이상 포함되어 있는 row의 수 출력 코드
print("결측치를 포함하고 있는 row 수 : ",train.isna().any(axis=1).sum())

train.Age = train['Age'].fillna(train['Age'].median())
train = train.drop('Cabin', axis=1)
train.Embarked = train['Embarked'].fillna('C')

train = pd.get_dummies(train, columns = ['Sex', 'Embarked'])

drop_cols = ['PassengerId', 'Name', 'Ticket']
train = train.drop(drop_cols, axis=1)
train

from sklearn.model_selection import train_test_split

# feature vector
X = train.drop("Survived", axis=1)
# target value
y = train["Survived"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state=42)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

import sklearn
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_features="sqrt", max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# 학습한 모델을 평가
from sklearn.metrics import accuracy_score

print("Train ACC : %.4f" %accuracy_score(y_train, clf.predict(X_train)))
print("Validation ACC : %.4f" %accuracy_score(y_val, clf.predict(X_val))
      
test.Fare = test['Fare'].fillna(test['Fare'].mean())
test.Age = test['Age'].fillna(test['Age'].median())
test = test.drop('Cabin', axis=1)
test.Embarked = test['Embarked'].fillna('C')
drop_cols = ['PassengerId', 'Name', 'Ticket']
test = pd.get_dummies(test, columns = ['Sex', 'Embarked'])
test = test.drop(drop_cols, axis=1)
X_test = test
output = clf.predict(X_test)
assert len(output) == 418

submission = pd.read_csv("gender_submission.csv")
submission.Survived = output
submission.to_csv('submission.csv', index=False)