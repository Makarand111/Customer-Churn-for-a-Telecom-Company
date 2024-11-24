# **Predicting Customer Churn for a Telecom Company**


# **Objective:**

## The objective is to predict customer churn for a telecom company using machine learning, enabling proactive measures to retain customers, enhance satisfaction, and optimize business strategies.

# **Import Data:**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

url = "https://github.com/YBIFoundation/Dataset/raw/8fc1128e8a4a6e0e01740ebd7ff10da5ee3b4b15/TelecomCustomerChurn.csv"
data = pd.read_csv(url)
data.head()

data.info()

data.describe()

data.columns

data.shape

# **Data Preprocessing:**

## Handling Missing Data:
#Check for missing values in the dataset and handle them appropriately ( either by imputing ar dropping)

data.isnull().sum()
# Select only numeric columns for median calculation
numeric_data = data.select_dtypes(include=np.number)

# Calculate the median for numeric columns only
median_values = numeric_data.median()

# Fill NaN values in the original DataFrame using the calculated medians
data.fillna(median_values, inplace=True)

## Encoding Categorical Variables:
#Convert the Categorical Columns into numerical form using techniques like One-hot Encoding or Label Encoding

data=pd.get_dummies(data,drop_first=True)

## Feature Engineering:
#Standardize or normalize the data for features that are on different scales.

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
data[['Tenure']] = scaler.fit_transform(data[['Tenure']])

# **Define Target (y) & Feature (x):**

x=data.drop('Churn_Yes',axis=1)
y=data['Churn_Yes']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x

y

x_train.shape

x_test.shape

y_train.shape

y_test.shape

# **Modeling: Model Selction**

from sklearn.linear_model import LogisticRegression

model=LogisticRegression(max_iter=1000)

model.fit(x_train,y_train)

# **Prediction:**

y_pred = model.predict(x_test)
y_pred
y_pred_prob = model.predict_proba(x_test)
y_pred_prob

# **Evaluation:**

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
print("Acuracy: ",accuracy)
print("Precision: ",precision)
print("Recall: ",recall)
print("F1_score: ",f1)
print("Confusion_Matrix: ",cm)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

### Confusion Matrix
### Displays the breakdown of true positives, true negatives, false positives, and false negatives in the prediction results.

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
plot_confusion_matrix(y_test, y_pred)

###ROC (Receiver Operating Characteristic) Curve
###Shows the trade-off between the True Positive Rate (TPR) and False Positive Rate (FPR), with the AUC score (Area Under the Curve) summarizing the model’s ability to distinguish between classes.

def plot_roc_curve(y_test, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])  # Probability for positive class
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
plot_roc_curve(y_test, y_pred_prob)

# **Conclusion & Accuracy Score:**

## The model achieved an accuracy of 82.18%, indicating it correctly predicts churn for the majority of customers. This suggests the model is effective in identifying overall patterns in the data.

# **Key features:**

### 1. Customers with shorter tenures are more likely to churn, as they may not have established loyalty to the company.
### 2. Higher monthly charges can lead to dissatisfaction, increasing the likelihood of churn.
### 3. Customers with month-to-month contracts are more prone to churn compared to those with longer-term contracts.
### 4. Frequent outages, unresolved complaints, or dissatisfaction with add-on services (e.g., internet or streaming) significantly impact churn.
### 5. Demographics such as age and location influence churn based on the company’s service reach and market competition.
### 6. The model can guide targeted retention campaigns focusing on high-risk customers with low tenure and high charges.
### 7. Decent precision helps avoid unnecessary expenditure on retaining low-risk customers.
### 8. Real-time churn prediction can enable quicker responses, improving customer satisfaction and retention.

