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

n_estimators — the number of decision trees you will be running in the model
criterion — this variable allows you to select the criterion (loss function) used to determine model outcomes.
 We can select from loss functions such as mean squared error (MSE) and mean absolute error (MAE).
  The default value is MSE.
max_depth — this sets the maximum possible depth of each tree
max_features — the maximum number of features the model will consider when determining a split
bootstrap — the default value for this is True, meaning the model follows bootstrapping principles (defined earlier)
max_samples — This parameter assumes bootstrapping is set to True, if not, this parameter doesn’t apply.
 In the case of True, this value sets the largest size of each sample for each tree.

Other important parameters are min_samples_split, min_samples_leaf, n_jobs, and
 others that can be read in the sklearn’s RandomForestRegressor documentation here.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('housing.csv',
                 usecols=['id', 'url', 'region', 'region_url', 'price', 'type', 'sqfeet', 'beds', 'baths',
                          'cats_allowed', 'dogs_allowed', 'smoking_allowed', 'wheelchair_access',
                          'electric_vehicle_charge', 'comes_furnished', 'laundry_options', 'parking_options',
                          'image_url', 'description', 'lat', 'long', 'state'])

df = df.drop(['id', 'url', 'region_url', 'image_url', 'description', 'state'], axis=1)
print(df.head())

# DATA CLEANING
df['laundry_options'] = df['laundry_options'].fillna(df['laundry_options'].mode()[0])
df['parking_options'] = df['parking_options'].fillna(df['parking_options'].mode()[0])
df['lat'] = df['lat'].fillna(df['lat'].mean())
df['long'] = df['long'].fillna(df['long'].mean())
df["baths"] = df["baths"].astype("int")

# remove outliers
outlier1 = ((df["beds"] > 4) | (df["baths"] > 4))
print(f"There is {df[outlier1]['beds'].count()} beds outliers.")
df = df[~outlier1]

outlier2 = ((df["sqfeet"] < 120) | (df["sqfeet"] > 5000) | (df["price"] < 100) | (df["price"] > 10000))
print(f"There is {df[outlier2]['cats_allowed'].count()} cats_allowed outliers.")
df = df[~outlier2]

df.describe()

df = df.drop(["cats_allowed"], axis=1)
df.rename(columns={'dogs_allowed': 'pets_allowed'}, inplace=True)

# Label Encoding categorical string values
le = LabelEncoder()
db = df
db["region"] = le.fit_transform(df["region"])
db["type"] = le.fit_transform(df["type"])
db["laundry_options"] = le.fit_transform(df["laundry_options"])
db["parking_options"] = le.fit_transform(df["parking_options"])
db.head()

# Split the data into features (X) and target (y)
X = db.drop('price', axis=1)
y = df['price']

# Multicollinearity heatmap
corrl = db.corr()
plt.figure(figsize=(20, 20))
heatmap = sns.heatmap(corrl, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 12},
                      cmap='twilight_shifted_r')
fig = heatmap.get_figure()
fig.savefig("heatmap.png")

scalar = StandardScaler()
x_scaled = scalar.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.30, random_state=470)

rf = RandomForestRegressor(n_estimators=90)
rf = rf.fit(X_train, y_train)
print(f"Score: {rf.score(X_test, y_test)}")

y_pred = rf.predict(X_test)

print(f'R^2: {metrics.r2_score(y_test, y_pred)}')
print(
    f'Adjusted R^2:{1 - (1 - metrics.r2_score(y_test, y_pred)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1)}')
print(f'MAE: {metrics.mean_absolute_error(y_test, y_pred)}')
print(f'MSE: {metrics.mean_squared_error(y_test, y_pred)}')
print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')
