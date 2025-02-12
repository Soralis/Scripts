import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, Lars, LassoLars, OrthogonalMatchingPursuit, PassiveAggressiveRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define the list of regression models
models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge()),
    ('Lasso Regression', Lasso()),
    ('ElasticNet Regression', ElasticNet()),
    ('Bayesian Ridge Regression', BayesianRidge()),
    ('Lars Regression', Lars()),
    ('Random Forest Regressor', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('Support Vector Regressor', SVR()),
    ('Gradient Boosting Regressor', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('K-Nearest Neighbors Regressor', KNeighborsRegressor(n_neighbors=5)),
    ('Decision Tree Regressor', DecisionTreeRegressor(random_state=42)),
    ('AdaBoost Regressor', AdaBoostRegressor(n_estimators=100, random_state=42)),
    ('MLP Regressor', MLPRegressor(random_state=42)),
    ('ElasticNet Regression', ElasticNet(random_state=42)),
    ('BayesianRidge Regression', BayesianRidge()),
    ('LassoLars Regression', LassoLars()),
    ('OrthogonalMatchingPursuit', OrthogonalMatchingPursuit()),
    ('PassiveAggressiveRegressor', PassiveAggressiveRegressor()),
]

# Define the list of classification models
classification_models = [
    ('Logistic Regression', LogisticRegression()),
    ('SVC', SVC()),
    ('LinearSVC', LinearSVC()),
    ('KNeighbors Classifier', KNeighborsClassifier()),
    ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
    ('Random Forest Classifier', RandomForestClassifier(random_state=42)),
    ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=42)),
    ('GaussianNB', GaussianNB()),
    ('BernoulliNB', BernoulliNB()),
    ('MLPClassifier', MLPClassifier(random_state=42)),
    ('AdaBoostClassifier', AdaBoostClassifier(random_state=42))
]

# Load your data (replace 'your_data.csv' with your actual data file)
data_dir = 'C:/Users/chris/OneDrive/Projekte/IFA/Scripts/A1/ball_size_ml/data'
pickle_path = os.path.join(data_dir, 'combined_data.pkl')

with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

# Preprocess the data
X = data.drop('Op40ForceAverage', axis=1).drop('Diameter1_4Diff', axis=1).drop('Diameter2_5Diff', axis=1).drop('Diameter3_6Diff', axis=1).drop('DiameterAvgDiff', axis=1)  # Replace 'target' with your target column name
y = data['Op40ForceAverage']

# Determine if the target variable is continuous or categorical
if pd.api.types.is_numeric_dtype(y):
    # Continuous target variable (regression)
    print("Target variable is continuous (regression)")
    models_to_evaluate = models
    metric = mean_squared_error
    metric_name = 'RMSE'
else:
    # Categorical target variable (classification)
    print("Target variable is categorical (classification)")
    models_to_evaluate = classification_models
    metric = accuracy_score
    metric_name = 'Accuracy'

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets (using a small subset for demonstration)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.98, random_state=42)  # 98% test, 2% train

# Train and evaluate each model
results = []
names = []

for name, model in models_to_evaluate:
    try:
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        if metric_name == 'RMSE':
            score = np.sqrt(metric(y_test, y_pred))
        else:
            score = metric(y_test, y_pred)
        results.append(score)
        names.append(name)

        # Print the results
        print(f'{name}: {metric_name} = {score}')
    except Exception as e:
        print(f'{name}: Error = {e}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.bar(names, results)
plt.xlabel('Model')
plt.ylabel(metric_name)
plt.title('Model Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
