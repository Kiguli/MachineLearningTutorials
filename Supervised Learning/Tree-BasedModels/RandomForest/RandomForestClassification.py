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
# Data Processing
import pandas as pd
import matplotlib.pyplot as plt
# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

bank_data = pd.read_csv('bank-additional-full.csv', sep=';', quotechar='"')
bank_data = bank_data.loc[:, ['age', 'default', 'cons.price.idx', 'cons.conf.idx', 'y']]

bank_data['default'] = bank_data['default'].map({'no': 0, 'yes': 1, 'unknown': 0})
bank_data['y'] = bank_data['y'].map({'no': 0, 'yes': 1})
print(bank_data)

# Split the data into features (X) and target (y)
X = bank_data.drop('y', axis=1)
y = bank_data['y']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit features to default RandomForest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# check accuracy of predictions
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# HYPERPARAMETER TUNING
param_dist = {'n_estimators': randint(50, 500),
              'max_depth': randint(1, 20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf,
                                 param_distributions=param_dist,
                                 n_iter=5,
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:', rand_search.best_params_)

# CONFUSION METRICS
# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot()
cm_display.figure_.savefig('confusion_matrix.png')

# accuracy, precision and recall stats
y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# FEATURE IMPORTANCE
# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot feature importances
fig, ax = plt.subplots(figsize=(10, 10))
feature_importances.plot.bar(ax=ax)
ax.set_title('Feature Importances')

# Save plot to file
fig.savefig('feature_importances.png')
