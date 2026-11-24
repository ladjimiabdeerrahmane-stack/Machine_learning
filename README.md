Hello
This is "Ladjimi Abde Errahmene" and this repository about learning the module "Machine Learning" :

# Python for Data Science & Machine Learning: Easy Tutorial

A simple, beginner-friendly tutorial covering the basics of Python data tools and the main machine learning methods.

## Who is this for?
Students and beginners who want to understand machine learning without complicated theory

## What you will learn

### 1. Python Tools
- **Python Basics**: variables, loops, functions
- **NumPy**: arrays and math operations
- **Pandas**: reading and handling data (CSV, tables)
- **Matplotlib**: plotting and visualizing data

### 2. Machine Learning
- **Supervised learning**
  - Linear Regression ✅
  - Logistic Regression ✅
  - KNN ✅
  - Decision Trees ✅
  - Random Forest (in progress)
  - SVM (in progress)
- **Unsupervised learning**
  - K-Means (in progress)
  - PCA (in progress)
- **Scikit-learn**: building models in a few lines
- **Model Evaluation**: MSE, R², accuracy

## How each lesson works
- Short and simple explanation
- Math step by step
- Python code you can run

## Requirements
- Python 3.8+
- Install the libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Example

```python
from sklearn.linear_model import LinearRegression

X = [[1], [3], [7], [10]]
y = [4, 13, 2, 7]

model = LinearRegression()
model.fit(X, y)
print(model.predict([[5]]))
```
