"""Random forests are a popular supervised machine learning algorithm.

- Random forests are for supervised machine learning, where there is a labeled target variable.
- Random forests can be used for solving regression (numeric target variable)
    and classification (categorical target variable) problems.
- Random forests are an ensemble method, meaning they combine predictions from other models.
- Each of the smaller models in the random forest ensemble is a decision tree.

In a random forest classification, multiple decision trees are created using different random subsets of the data
 and features. Each decision tree is like an expert, providing its opinion on how to classify the data.
 Predictions are made by calculating the prediction for each decision tree, then taking the most popular result.
  (For regression, predictions use an averaging technique instead.)
"""
