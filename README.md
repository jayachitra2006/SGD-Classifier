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
df=pd.DataFrame(data=iris.data,columns=ir
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

