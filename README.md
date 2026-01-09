# Diabetes Progression Prediction using Machine Learning

## Project Overview
This project explores how machine learning models behave under different levels of complexity using a real-world diabetes dataset.  
The main focus is to understand **underfitting**, **overfitting**, and the **bias–variance tradeoff**.

---

## Dataset
- Source: Built-in Diabetes dataset from scikit-learn
- Number of samples: 442 patients
- Number of features: 10 medical attributes
- Target: Quantitative measure of diabetes disease progression

---

## Machine Learning Approach
The project uses **Linear Regression** with **Polynomial Feature Expansion** to simulate models of increasing complexity.

Polynomial degrees tested:
- Degree 1 → Simple model (Underfitting)
- Degree 3 → Balanced model
- Degree 10 → Complex model (Overfitting)

---

## Evaluation Metric
- **Mean Squared Error (MSE)** was used to evaluate model performance.
- Performance was compared on:
  - Training data
  - Testing (unseen) data

---

## Key Observations
- Low-degree models underfit the data and perform poorly on both training and test sets.
- Medium-degree models achieve a good balance between bias and variance.
- High-degree models overfit the training data and perform poorly on test data.
- Visualization clearly demonstrates the bias–variance tradeoff.

---

## Visualization
The project includes a plot comparing training and testing errors across different model complexities to visually demonstrate overfitting and underfitting.

---

## Tools & Libraries Used
- Python
- scikit-learn
- NumPy
- Matplotlib
- PyCharm

---

## What I Learned
- How model complexity affects generalization
- Practical understanding of underfitting and overfitting
- Importance of evaluating models on unseen data
- How to structure and present an ML project professionally

---

## Future Improvements
- Add cross-validation for more robust evaluation
- Experiment with regularization techniques
- Extend the project to other regression models
