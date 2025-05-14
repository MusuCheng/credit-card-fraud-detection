import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

file_path = "creditcard.csv" 
df = pd.read_csv(file_path)

scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]

conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
auprc = auc(recall, precision)

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)
print(f"\nAUPRC: {auprc:.4f}")