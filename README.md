# 🚗🔍 Vehicle Insurance Fraud Claim  Detection

## 📌 Project Overview  
Vehicle insurance fraud occurs when individuals collaborate to deceive or overstate the extent of damage or injuries resulting from a car accident. This project utilizes **machine learning techniques** to detect fraudulent claims effectively.

## 📂 Dataset  
📌 **Source:** [Kaggle](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)  
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

## 📊 Results  

### **Classifier Accuracies Before Feature Selection:**  
| Model              | Accuracy (%) |
|--------------------|-------------|
| Random Forest (RF) | **89.6%**    |
| Decision Tree (DT) | 78.7%        |
| Support Vector Machine (SVM) | 76.52%  |
| Naïve Bayes       | 77.59%       |
| Logistic Regression | 77.02%     |

### **Accuracies After Feature Selection (Mutual Information - 17 features):**  
| Model             | Accuracy (%) |
|-------------------|-------------|
| Random Forest    | **90.47%**    |
| Decision Tree    | 80.33%        |

### **Accuracies After Feature Selection (Mean Decrease Gini Impurity - 19 features):**  
| Model            | Accuracy (%) |
|------------------|-------------|
| Random Forest   | **90.83%**    |
| Decision Tree   | 79.98%        |
| SVM             | 76.13%        |

### **Accuracies Using 14 Common Features in the Shiny App:**  
| Model           | Accuracy (%) |
|----------------|-------------|
| Random Forest  | **87.74%**   |
| SVM           | 79.10%        |
| Decision Tree  | 74.88%        |

📌 **Best Model:** **Random Forest** consistently achieved the highest accuracy of **87.74%** using the **14 common features in the Shiny app**.  

📌 **AUC Scores:**  
- **Random Forest (Before Feature Selection):** **0.91**  
- **Random Forest (Shiny App - 14 common features):** **0.959** (+0.04 improvement)  

## 🖥️ Shiny App  
A **Shiny app** was developed to predict whether an insurance claim is **Fraudulent** or **Not Fraudulent** using the **14 common features**. The app utilizes the **Random Forest model**, achieving **high accuracy and AUC value of 0.959**.  

## 🔹 How It Works?
### 📝 User Inputs Claim Details  
The app provides an interactive form where users enter details such as:  
- **Vehicle Age**  
- **Policy Number**  
- **Claim Amount**  
- **Driver Rating**  
- **Policy State**  
- **Month of Claim**  
- **Days to Claim Reporting**  
- **Accident Severity**  
- **Make of the Vehicle**  
- **Vehicle Price**  
- **Age of Policyholder**  
- **Past Claims**  
- **Number of Witnesses**  
- **Police Report Filed** (Yes/No)  
 

### ⚡ Model Prediction  
- The **Random Forest model** processes the inputs and predicts whether the claim is **fraudulent (1) or not (0)**.  

### 📊 Results Visualization  
- The app displays the **fraud probability** along with key **contributing factors**.  

## 🔹 Technology Stack  

- **Frontend:** Shiny (R-based UI framework)  
- **Backend:** R (Machine Learning with `randomForest` and `glmnet`)  
- **Data Handling:** `tidyverse`, `dplyr`, `caret`  
- **Deployment:** Hosted on **ShinyApps.io**  

### 🛠 **Shiny App Dependencies**  
Ensure the following R packages are installed:  
```sh
install.packages(c("shiny", "randomForest", "shinythemes", "dplyr", "ggplot2", "caret", "ROSE"))
```

## 🚀 Installation & Setup  

### 🔹 **1. Clone the Repository**  
```sh
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
﻿
