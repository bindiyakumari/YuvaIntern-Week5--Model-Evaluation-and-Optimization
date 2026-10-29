# Week 5: Model Evaluation and Optimization

This repository contains my Week 5 project for the Data Science with Python Virtual Internship. The objective of this task is to perform rigorous model evaluation and optimization on a highly imbalanced classification problem.

## 📊 Dataset Overview
*   **Dataset:** Credit Card Fraud Detection (`creditcard.csv`)
*   **Challenge:** Extreme class imbalance (fraudulent transactions represent only ~0.17% of the data).
*   **Objective:** Maximize the detection of fraudulent transactions (Recall) while maintaining a reasonable rate of false alarms (Precision).

## 🛠️ Key Methodologies Implemented
1.  **Preventing Data Leakage:** Implemented `Scikit-Learn` Pipelines to ensure `StandardScaler` is fitted only on training data after the train-test split.
2.  **Imbalanced Metrics:** Shifted evaluation focus away from Accuracy and ROC-AUC toward **Precision-Recall Curves** and **Average Precision (AP)**, which are far more reliable for imbalanced datasets.
3.  **Cross-Validation:** Used `StratifiedKFold` to maintain the rare fraud distribution across all training folds.
4.  **Model Comparison:** Evaluated and compared multiple models including:
    *   Baseline Logistic Regression
    *   Class-Weighted Logistic Regression
    *   Decision Tree Classifier
    *   Random Forest Classifier
5.  **Hyperparameter Tuning:** Conducted a grid search (`GridSearchCV`) to find the optimal regularization parameter for Logistic Regression.
6.  **Threshold Tuning (Leakage-Free):** Extracted out-of-fold predictions on the *training set* using `cross_val_predict` to find the optimal probability threshold, ensuring the test set remained completely untouched until the final evaluation.

## 📈 Key Findings
*   **Baseline Weakness:** The baseline Logistic Regression model suffered from poor recall, missing a significant portion of fraudulent transactions.
*   **Class Weighting Trade-off:** Adding `class_weight='balanced'` drastically improved recall but caused a severe drop in precision (too many false positives).
*   **Optimal Solution:** The **Random Forest Classifier** with a manually tuned probability threshold achieved the highest Average Precision. It provided the best overall F1-score on the final test set, successfully balancing the trade-off between catching fraud and blocking legitimate customers.

## 💻 Technologies Used
*   Python 3
*   Pandas (Data Manipulation)
*   Scikit-Learn (Modeling, Pipelines, Cross-Validation, Metrics)
*   Matplotlib (Data Visualization)
