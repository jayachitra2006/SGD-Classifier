# ML-EXP-7
# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Iris dataset and convert it into a Pandas DataFrame.

2.Separate the dataset into independent variables (X) and dependent variable (Y).

3.Split the data into training and testing sets using the train–test split method.

4.Create the SGD Classifier model with required parameters.

5.Train the model using the training data and predict the class labels for test data.

6.Evaluate the model performance using accuracy score and confusion matrix.
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: JAYACHITRA J
RegisterNumber: 212224040132
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df.head())
X=df.drop('target',axis=1)
y=df['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Acuuracy:{accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show() 

```

## Output:
<img width="1255" height="504" alt="Screenshot 2026-02-06 113037" src="https://github.com/user-attachments/assets/961e960f-4b7b-42a0-a7d9-fc2c805ffa5b" />
<img width="686" height="512" alt="image" src="https://github.com/user-attachments/assets/c798ec3c-4eda-4797-84df-4ed9009c07ef" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

