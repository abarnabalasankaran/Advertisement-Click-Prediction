# 📢 Advertisement Click Prediction

This machine learning project focuses on predicting whether a user will click on an online advertisement based on their personal and browsing information. The goal is to build and compare multiple classification models and determine which performs best for this classification task.

---

## 🧠 Project Summary

Machine Learning has become integral in automating decisions and driving business value. This project applies machine learning to a real-world business scenario—predicting advertisement click-through based on user data. The task involves building classification models to predict user behavior, optimizing accuracy, and generating actionable insights from the data.

---

## 🎯 Project Objective

To develop a machine learning model that classifies whether a user will click on an ad based on features like age, income, gender, time spent on the website, and other behavior-based variables.

---

## 🔧 Steps Performed

1. **Imported Required Libraries**  
   Refer to `requirements.txt` for all dependencies.

2. **Data Preprocessing**  
   - Checked for missing values  
   - Encoded categorical variables  
   - Normalized/standardized data  

3. **Model Building & Evaluation**  
   Trained and evaluated the following models:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - Naive Bayes
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Gradient Boosting
   - AdaBoost Classifier
   - Multi-layer Perceptron (MLP/ANN)
   - XGBoost Classifier

4. **Visualization**  
   - Used `seaborn.pairplot` and heatmaps for EDA

5. **Model Comparison**  
   - Accuracy scores calculated and models ranked accordingly

---

## 📈 Accuracy Comparison

| Model                         | Accuracy Score |
|------------------------------|----------------|
| KNN Algorithm                 | **0.93**       |
| MLP Classifier (ANN)         | **0.93**       |
| Random Forest Classifier     | 0.92           |
| Support Vector Machine       | 0.92           |
| Naive Bayes Algorithm        | 0.88           |
| XGBoost Classifier Algorithm | 0.89           |
| Gradient Boosting            | 0.89           |
| Logistic Regression          | 0.87           |
| Decision Tree Classifier     | 0.86           |
| AdaBoost Classifier          | 0.86           |

---

## 🏆 Best Performing Models

1. **K-Nearest Neighbors (KNN)**
2. **MLP Classifier (Artificial Neural Network)**
3. **Random Forest Classifier**
4. **Support Vector Machine**
5. **Gradient Boosting**

---

## 🛠️ Libraries Used

- **NumPy** – Numerical computations
- **Pandas** – Data manipulation
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning models and preprocessing
- **XGBoost** – Gradient boosting framework


