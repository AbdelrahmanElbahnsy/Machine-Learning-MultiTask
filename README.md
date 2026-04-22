# 🚀 Machine Learning Multi-Task Portfolio

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=flat&logo=scikit-learn)
![Gradio](https://img.shields.io/badge/Gradio-UI-red?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
A comprehensive Machine Learning portfolio showcasing four distinct predictive models covering both **Classification** and **Regression** tasks. All models are unified into a single, interactive, and user-friendly web dashboard built with **Gradio**.

This project demonstrates practical end-to-end ML skills, including data preprocessing, feature engineering, model optimization, and deployment.

---

## 🛠️ The 4 Machine Learning Tasks

### 🌸 1. Iris Flower Classification
* **Objective:** Classify Iris flowers into 3 species based on sepal and petal dimensions.
* **Algorithm:** Decision Tree Classifier (`DecisionTreeClassifier`).
* **Performance:** 🎯 **100% Accuracy**.
* **Highlights:** Demonstrates basic categorical classification and rule-based decision-making.

### ❤️ 2. Heart Disease Predictor
* **Objective:** Predict the likelihood of heart disease based on patient medical metrics.
* **Algorithm:** K-Nearest Neighbors (`KNeighborsClassifier`).
* **Performance:** 🎯 **86% Accuracy**.
* **Highlights:** Utilized `StandardScaler` to normalize distance metrics, ensuring stable and accurate KNN predictions.

### 🛡️ 3. SMS Spam Detector
* **Objective:** Identify and filter out spam/promotional messages from normal texts.
* **Algorithm:** Multinomial Naive Bayes (`MultinomialNB`).
* **Performance:** 🎯 **90% Accuracy**.
* **Highlights:** Applied **NLP** techniques using `TF-IDF Vectorizer` to extract feature weights and ignore irrelevant stop words.

### 💵 4. Medical Insurance Cost Predictor
* **Objective:** Predict individual medical insurance costs based on age, BMI, children, and smoking habits.
* **Algorithm:** Polynomial Linear Regression.
* **Performance:** 🎯 **86.6% R2 Score**.
* **Highlights:** Upgraded from a simple linear model to `PolynomialFeatures` to capture non-linear relationships (e.g., the compounding effect of smoking and aging), drastically improving the model's accuracy.

---

## 💻 Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Deployment/UI:** Gradio

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ahmedabdoahmed007/ML-Tasks-Portfolio.git](https://github.com/ahmedabdoahmed007/ML-Tasks-Portfolio.git)
   cd ML-Tasks-Portfolio
