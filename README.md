# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries, load the Employee dataset, and encode categorical values such as salary into numerical form.
2. Select employee details as input features (X) and the employee churn column (left) as the target variable (y). Split the dataset into training and testing sets.
3. Create the Decision Tree Classifier using the entropy criterion and train the model using the training dataset.
4. Predict employee churn for test data, calculate the accuracy of the model, and visualize the decision tree using plot_tree().

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Rougith D
RegisterNumber: 212225220087
*/

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
data = {
    "Age": [22, 35, 45, 28, 50, 41, 30, 26, 48, 33],
    "Salary": [20000, 50000, 70000, 30000, 90000, 65000, 40000, 25000, 85000, 48000],
    "YearsAtCompany": [1, 7, 15, 2, 20, 10, 4, 1, 18, 6],
    "JobSatisfaction": [2, 4, 3, 2, 5, 4, 3, 1, 5, 3],
    "Churn": ["Yes", "No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "No"]
}

df = pd.DataFrame(data)
X = df[["Age", "Salary", "YearsAtCompany", "JobSatisfaction"]]
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=["Age", "Salary", "YearsAtCompany", "JobSatisfaction"],
    class_names=["No", "Yes"],
    filled=True
)
plt.title("Decision Tree Classifier for Employee Churn Prediction")
plt.show()
```

## Output:
<img width="701" height="235" alt="image" src="https://github.com/user-attachments/assets/01c846cf-cc5c-4ddc-9bed-7230ddfbf7ae" />
<img width="953" height="787" alt="image" src="https://github.com/user-attachments/assets/3c0da78f-cab5-471d-9d7d-f75268dab157" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
