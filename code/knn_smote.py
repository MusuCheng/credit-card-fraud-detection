import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from collections import Counter


# Load dataset
file_path = "creditcard.csv"
df = pd.read_csv(file_path)

# Scale 'Time' and 'Amount'
scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

# Split features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Train/test split (keep test set clean)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Original training set: {Counter(y_train)}")
print(f"After SMOTE: {Counter(y_train_res)}")

# Sample for tuning
X_sample, _, y_sample, _ = train_test_split(
    X_train_res, y_train_res, train_size=5000, stratify=y_train_res, random_state=42
)

# K tuning on SMOTE’d sample
k_range = range(1, 11)
auprc_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_sample, y_sample)
    y_proba = knn.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    score = auc(recall, precision)
    auprc_scores.append(score)

# Pick best k
best_k = k_range[np.argmax(auprc_scores)]
print(f"Best k (SMOTE sample): {best_k} (AUPRC = {max(auprc_scores):.4f})")

# Final model trained on full SMOTE’d training set
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_res, y_train_res)

# Predictions
y_pred = knn_final.predict(X_test)
y_proba = knn_final.predict_proba(X_test)[:, 1]

# Final evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
auprc = auc(recall, precision)

print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)
print(f"\nFinal AUPRC: {auprc:.4f}")

# Plot AUPRC vs. k
plt.figure(figsize=(10, 5))
plt.plot(k_range, auprc_scores, marker='o')
plt.title('AUPRC vs. K (after SMOTE)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('AUPRC')
plt.grid(True)
plt.tight_layout()
plt.show()
