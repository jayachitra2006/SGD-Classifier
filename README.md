# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1: Start
STEP 2: Load the Iris dataset and create a Pandas DataFrame with features and target.
STEP 3: Split the dataset into features (X) and target (y).
STEP 4: Split the data into training and testing sets using train_test_split.
STEP 5: Initialize the SGDClassifier with default parameters.
STEP 6: Train the classifier using the training data.
STEP 7: Predict the target values for the testing set.
STEP 8: Calculate and display the model's accuracy and confusion matrix.
STEP 9: End 

## Program:
```
# ---------------------------------------------------
# Prediction using SGD Classifier
# Developed by: J.Jayachitra
# Register Number: 212224040132
# ---------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# Step 1: Sample Dataset
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Previous_Score': [40, 50, 55, 60, 65, 70, 75, 80],
    'Internship': [0, 0, 1, 0, 1, 1, 1, 1],  # 0 = No, 1 = Yes
    'Placement': [0, 0, 0, 1, 1, 1, 1, 1]    # Target: 0 = Not Placed, 1 = Placed
}
df = pd.DataFrame(data)
# Step 2: Split Features & Target
X = df[['Hours_Studied', 'Previous_Score', 'Internship']]
y = df['Placement']
# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
# Step 4: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Step 5: Train SGD Classifier
sgd_model = SGDClassifier(
    loss='log_loss',          # Logistic Regression
    max_iter=1000,
    learning_rate='optimal',
    random_state=42
)
sgd_model.fit(X_train, y_train)
# Step 6: Predictions
y_pred = sgd_model.predict(X_test)
y_prob = sgd_model.predict_proba(X_test)
# Step 7: Model Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Step 8: New Student Prediction
new_student = np.array([[6, 68, 1]])  # Example input
new_student_scaled = scaler.transform(new_student)
placement_pred = sgd_model.predict(new_student_scaled)
placement_prob = sgd_model.predict_proba(new_student_scaled)
print("\nPredicted Placement Status:",
      "Placed" if placement_pred[0] == 1 else "Not Placed")
print("Probability of Placement:", round(placement_prob[0][1], 2))
```

## Output:
<img width="734" height="570" alt="Screenshot 2026-03-13 083036" src="https://github.com/user-attachments/assets/aed1e811-2a19-4c40-97ec-2791ed640066" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
