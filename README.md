# Machine-Learning-Algorithms
Core machine learning algorithms implemented from scratch in Python — Linear Regression, Multiple Linear Regression, Logistic Regression, ID3 Decision Trees, and K-Means Clustering — built without sklearn to understand the underlying math. Implemented using NumPy, Pandas, Matplotlib, and Seaborn.

# Machine Learning Algorithms from Scratch 

A collection of core ML algorithms implemented **from scratch in Python** — no `sklearn.fit()` shortcuts. Each notebook builds the math manually: cost functions, gradients, entropy calculations, and distance-based clustering, so I could understand exactly how these models learn instead of treating them as black boxes.

## Libraries & Tools Used

- **NumPy** – core numerical operations, matrix math, gradient calculations
- **Pandas** – data loading, cleaning, and preprocessing
- **Matplotlib** – plotting cost curves, regression lines, elbow plots, convergence graphs
- **Seaborn** – confusion matrix heatmaps and styled visualizations
- **Python's `math` module** – log base 2 calculations for entropy (used in the decision tree)
- **Google Colab** – development environment, with datasets loaded from Google Drive / public URLs

No `scikit-learn`, `TensorFlow`, or `PyTorch` was used for the core algorithm logic — every model (cost function, gradients, splitting criteria, clustering steps) was coded manually.

## What's inside

### 1. Linear Regression (Simple)
**Dataset:** Real estate price vs. size

- Implemented the hypothesis function, Mean Squared Error cost function, and gradient descent manually
- Applied feature scaling (standardization) to stabilize gradient descent
- Visualized actual vs. predicted prices and the cost reduction curve over iterations

### 2. Multiple Linear Regression & Logistic Regression
**Dataset:** Housing prices

- Extended linear regression to multiple features using a bias-augmented design matrix
- Built logistic regression for binary classification (expensive vs. affordable housing) using the sigmoid function and binary cross-entropy loss
- Compared convergence behavior across different learning rates
- Evaluated with a manually built confusion matrix

### 3. ID3 Decision Tree (from scratch)
**Dataset:** UCI Mushroom Classification dataset

- Implemented Shannon entropy and Information Gain calculations by hand
- Built a recursive ID3 algorithm that picks the best splitting feature at each node
- Compared information gain across attributes (e.g. odor vs. bruises) to see which features split the data best
- Evaluated using a custom confusion matrix and accuracy score

### 4. K-Means Clustering (from scratch)
**Dataset:** Mall customer segmentation data

- Implemented the full K-Means loop manually: centroid initialization, cluster assignment (Euclidean distance), centroid updates, and convergence checking
- Used the Elbow Method to find the optimal number of clusters
- Visualized customer segments and interpreted each cluster's real-world meaning (e.g. high income/high spending = premium customers)

## Why I built these from scratch

It's easy to call `model.fit()` and get results without understanding what's actually happening underneath. These notebooks were my way of forcing myself to understand:
- How gradient descent updates parameters step by step
- How entropy and information gain decide tree splits
- How distance-based clustering groups unlabeled data
- Why feature scaling and preprocessing choices affect model performance

## Key takeaways

- Covered both supervised learning (regression, classification) and unsupervised learning (clustering) — implemented entirely from the ground up
- Learned how preprocessing decisions (handling missing values, encoding, scaling) directly affect model results
- Practiced evaluating models manually using confusion matrices, accuracy, and entropy-based metrics, without relying on library shortcuts

## Future Improvements

- Add pruning to the decision tree to reduce overfitting
- Benchmark from-scratch results against `sklearn` implementations
- Extend K-Means with silhouette score for more robust validation
- Package each algorithm as a reusable Python module/class

---

*Built as part of my Machine Learning coursework, with a focus on understanding the algorithms at the math level before relying on libraries.*
