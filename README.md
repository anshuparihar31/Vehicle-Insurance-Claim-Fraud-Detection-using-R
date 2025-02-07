# 🚗🔍 Vehicle Insurance Fraud Claim  Detection

## 📌 Project Overview  
Vehicle insurance fraud occurs when individuals collaborate to deceive or overstate the extent of damage or injuries resulting from a car accident. This project utilizes **machine learning techniques** to detect fraudulent claims effectively.

## 📂 Dataset  
📌 **Source:** Kaggle  
📌 **Total Rows:** 15,420  
📌 **Total Columns:** 33  
📌 **Numeric Features:** Age, Repnumber, Deductible, DriverRating  
📌 **Target Variable:** `FraudFound_P` (0 = No Fraud, 1 = Fraud)  

## 🛠 Data Preprocessing  
✔️ **Categorical Encoding:** Categorical variables are encoded as factors.  
✔️ **Missing Value Handling:** No explicit missing value handling in the dataset.  
✔️ **Oversampling:** The dataset is **oversampled using the ROSE package** to handle class imbalance.

## 📊 Feature Selection Methods  

🔹 **1. Mutual Information (MI)**  
- Measures the amount of information one variable provides about another.  
- Identifies the most relevant features for fraud detection.  
- Top **17 features** are selected for model training.

🔹 **2. Mean Decrease in Gini Impurity**  
- Used with **Random Forest & Decision Trees**.  
- Determines feature importance based on impurity reduction in trees.  

## 🤖 Algorithm Implementation  

### 🌳 **Decision Tree (`rpart`)**  
📌 **Method:** Classification (`method = "class"`)  
📌 **Hyperparameter:** `cp = 0.001` (controls tree complexity to avoid overfitting)  

### 🌲 **Random Forest (`randomForest`)**  
📌 **Hyperparameters:**  
✔️ `mtry = 6` (number of variables randomly sampled for each split)  
✔️ `ntree = 700` (number of trees in the forest)  
✔️ `nodesize = 5` (minimum terminal node size)  

### 🧮 **Naive Bayes**  
📌 **Hyperparameters:**  
✔️ Laplace smoothing (`.laplace`)  
✔️ Kernel density estimates (`.usekernel`)  
✔️ Bandwidth adjustment (`.adjust`)  

### 📈 **Logistic Regression (`glmnet`)**  
📌 **Hyperparameters:**  
✔️ `alpha = 0:1` (LASSO `alpha = 1` and Ridge `alpha = 0`)  
✔️ `lambda` values from **0.01 to 1** (controls regularization strength)  

### ✨ **Support Vector Machine (SVM)**  
📌 **Hyperparameter:** `C = 1` (controls the balance between error minimization and model complexity)  

## 🚀 Installation & Setup  

### 🔹 **1. Clone the Repository**  
```sh
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
﻿
