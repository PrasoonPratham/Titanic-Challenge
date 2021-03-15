import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
test_data = pd.read_csv("https://raw.githubusercontent.com/PrasoonPratham/Kaggle/main/titanic/test.csv")
train_data = pd.read_csv("https://raw.githubusercontent.com/PrasoonPratham/Kaggle/main/titanic/train.csv")
train_labels = train_data['Survived']
train_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True) 
train_data['Sex'] = train_data.Sex.map({"male":1, "female":0})
train_data['Embarked'] = train_data.Embarked.map({"C":0, "Q":1, "S" :2})
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].median())
trainData , valData, trainLabels, valLabels = (train_test_split(train_data, train_labels,train_size=0.76))
decision_tree = tree.DecisionTreeClassifier(max_depth = 3)
model = decision_tree.fit(trainData, trainLabels) 
test_data.drop([ 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True) 
test_data['Sex'] = test_data.Sex.map({"male":0, "female":1})
test_data['Embarked'] = test_data.Embarked.map({"C":0, "Q":1, "S" :2})
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
pred = model.predict(test_data)
pred = pred.astype(int)
submission = pd.read_csv("https://raw.githubusercontent.com/PrasoonPratham/Kaggle/main/titanic/gender_submission.csv")
submission['Survived'] = pred
submission.to_csv('submission.csv', index=False)