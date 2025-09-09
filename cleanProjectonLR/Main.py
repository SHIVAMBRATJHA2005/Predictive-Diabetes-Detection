# Logistic Regression new
# Data handling
import pandas as pd
import numpy as np

# Train-test split
from sklearn.model_selection import train_test_split

# Logistic Regression model
from sklearn.linear_model import LogisticRegression

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# (Optional) Visualization
import matplotlib.pyplot as plt
import seaborn as sns


url = "https://raw.githubusercontent.com/Om-Kumar-Ace/Diabetes-Health-Indicator/refs/heads/main/Data/diabetes_data.csv"

# Load directly into DataFrame
df = pd.read_csv(url)

# Save into your local Data/ folder
df.to_csv("Data/diabetes_data.csv", index=False)

print("Dataset saved to Data/diabetes_data.csv")


import os
import pandas as pd

# Make sure Data/ exists
os.makedirs("Data", exist_ok=True)

# Link for trained dataset
train_url = "https://raw.githubusercontent.com/Om-Kumar-Ace/Diabetes-Health-Indicator/refs/heads/main/final_diabetes_train.csv"

# Save trained dataset
train_df = pd.read_csv(train_url)
train_df.to_csv("Data/final_diabetes_train.csv", index=False)
print("✅ Trained dataset saved as Data/final_diabetes_train.csv")


import pandas as pd
import numpy as np

# Load the dataset
final_Dataset = pd.read_csv("Data/final_diabetes_train.csv")

# Split dataset into train and test (75%-25%)
final_Dataset["is_train"] = np.random.uniform(0, 1, len(final_Dataset)) <= 0.75

train = final_Dataset[final_Dataset["is_train"] == True]
test = final_Dataset[final_Dataset["is_train"] == False]

print("No. of observations for the training dataset:", len(train))
print("No. of observations for the test dataset:", len(test))

# Optional: list features (assuming last column is target)
features = final_Dataset.columns[:-2]
print("Features used for model:", features.tolist())

# 1. Import Required Modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load Dataset
df = pd.read_csv("Data/final_diabetes_train.csv")

# Features and Target
X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Logistic Regression Model
log_reg = LogisticRegression(
    solver='saga',       # saga works well for large datasets
    max_iter=2000,       # increased iterations for convergence
    n_jobs=-1,           # use all CPU cores for speed
    random_state=42
)

log_reg.fit(X_train_scaled, y_train)

# 6. Evaluation
y_pred = log_reg.predict(X_test_scaled)

print("Intercept:", log_reg.intercept_)
print("Coefficients:", log_reg.coef_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for class 1 (Diabetes=1)
y_score = log_reg.predict_proba(X_test_scaled)[:, 1]

# ROC curve
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_score)
roc_auc = auc(false_positive_rate, true_positive_rate)

# Plot ROC
plt.figure(figsize=(8,6))  # make the figure larger
plt.plot(false_positive_rate, true_positive_rate, 'b', 
         label="AUC = %0.3f" % roc_auc, linewidth=2)

plt.plot([0, 1], [0, 1], 'r--')  # diagonal reference line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR = FP / (FP+TN))')
plt.ylabel('True Positive Rate (TPR = TP / (TP+FN))')
plt.title('Receiver Operating Characteristic (ROC) - Diabetes Logistic Regression')
plt.legend(loc="lower right")

plt.savefig("roc_diabetes.png", dpi=300)  # ✅ save first, high resolution
plt.show()  # ✅ then show


from sklearn.model_selection import cross_val_score

# Cross-Validate Model Using Recall
recall_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring="recall")
print("Recall scores:", recall_scores)
print("Mean Recall:", recall_scores.mean())

# Cross-Validate Model Using Precision
precision_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring="precision")
print("Precision scores:", precision_scores)
print("Mean Precision:", precision_scores.mean())


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Predictions
y_pred = log_reg.predict(X_test_scaled)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC-AUC score
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
print("\nROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
