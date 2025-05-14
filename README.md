# Credit Card Fraud Detection

This project applies multiple machine learning techniques to detect fraudulent credit card transactions using a real-world, highly imbalanced dataset. The models include K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machines (SVM), Tree-based models, and a hybrid deep learning model built on signal decomposition techniques.

## 📊 Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Transactions made by European cardholders in September 2013.
- 284,807 transactions total, only 492 (0.172%) are fraud.

## 🧠 Models Used

| Model                      | Description |
|---------------------------|-------------|
| **Logistic Regression**   | Simple, interpretable baseline with high accuracy and precision. |
| **K-Nearest Neighbors (KNN)** | Evaluated with/without SMOTE, tuned on AUPRC. |
| **Support Vector Machines (SVM)** | Effective in high-dimensional space, strong AUC performance. |
| **Decision Trees**        | Compared weighted vs. unweighted variants. |
| **Random Forest**         | Recommended for better generalization. |
| **Hybrid Model (EEMD-MMRVFLN+)** | Combines ensemble empirical mode decomposition and multi-kernel networks for imbalanced classification. |

## 🔍 Techniques & Highlights

- **Class Balancing**: SMOTE used to oversample fraud cases.
- **AUPRC Optimization**: Used as the main metric instead of accuracy due to imbalance.
- **Threshold Tuning**: Custom F1-based threshold optimization in hybrid models.
- **Model Comparison**: Trade-offs between precision, recall, and AUC across all models.

## 📁 Project Structure

├── KNN.py # Basic KNN implementation
├── knn_smote.py # KNN with SMOTE and tuning
├── KNN_tuning.py # K tuning using AUPRC
├── logistic regression(Credit Card Fraud Detection).py
├── Support Vector Machines.ipynb # SVM modeling notebook
├── Tree Model.ipynb # Decision Tree evaluation
├── Data Mining Term Project.pptx # Final presentation
├── Term Project (Data Mining).docx # Final written report
├── README.md

less
Copy
Edit

## 🛠 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place `creditcard.csv` in your working directory.
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn imbalanced-learn matplotlib numpy
Run any Python script (e.g. KNN.py) using:

bash
Copy
Edit
python KNN.py
👥 Contributors
Keyu (Kevin) Chen – Logistic Regression, Project Integration

Niketan Baranwal – SVM, Tree Models

Debasis Tripathy – Hybrid Model (EEMD-MMRVFLN+)

Mark Grober – KNN modeling & SMOTE evaluation

📌 Key Takeaway
Even simple models like Logistic Regression and KNN can achieve strong fraud detection when combined with proper sampling and evaluation strategies. However, hybrid and ensemble approaches provide better balance and robustness in imbalanced, real-world financial data.