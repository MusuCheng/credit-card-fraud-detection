import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load data (ensure the CSV file is in your working directory)
data = pd.read_csv('creditcard.csv')

# Create balanced dataset
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
legit_sample = legit.sample(n=len(fraud), random_state=1)
balanced_data = pd.concat([legit_sample, fraud])

# Features and labels
X = balanced_data.drop(columns='Class')
y = balanced_data['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)  # Use max_iter=1000 for convergence
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_pred)

# Print results
print("Training Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC:", auc)