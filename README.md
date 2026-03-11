# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.

2. Load the dataset containing lab test results and tumor class labels (benign or malignant).

3. Preprocess the data by handling missing values and splitting the dataset into training and testing sets.

4. Train the Decision Tree model using the training dataset.

5. Test and evaluate the model using the testing dataset to classify tumors as benign or malignant and measure accuracy.

6. Display the classification results and stop the program.

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: Balaji B
RegisterNumber: 212225040040
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("tumor.csv")                                                    

print(data.head())
print(data.columns)

x=data.drop(columns=['Class'])
y=data['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Name:Balaji B")
print("Reg.No: 212225040040")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```


## Output:
![alt text](<Screenshot 2026-03-11 085929.png>)
![alt text](<Screenshot 2026-03-11 085954.png>)

## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
