import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
dataset = pd.read_csv("Weatherdata.csv")

# Extract features and target
x = dataset[['Temperature', 'Wind_Speed', 'UV_Index', 'Wind_Direction', 'Pressure', 'Humidity', 'Cloud_Cover']].values
y = dataset['RainTomorrow'].values

# Handle missing values using the most frequent strategy
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
x = imputer.fit_transform(x)
y = imputer.fit_transform(y.reshape(-1, 1)).ravel()

# Encode categorical features
le_wind_direction = LabelEncoder()
x[:, 3] = le_wind_direction.fit_transform(x[:, 3])  # Encode Wind Direction

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define the model
classifier = RandomForestClassifier(random_state=0)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# Best parameters from Grid Search
print(f"Best Parameters: {grid_search.best_params_}")

# Train the RandomForest model with best parameters
best_classifier = grid_search.best_estimator_

# Evaluate using Cross-Validation
cv_scores = cross_val_score(best_classifier, x, y, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")

# Make predictions
y_pred = best_classifier.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save the model, scaler, and encoder
joblib.dump(best_classifier, 'random_forest_rainfall_model.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')
joblib.dump(le_wind_direction, 'encoder_wind_direction.pkl')
joblib.dump(le_target, 'encoder_target.pkl')

# Output accuracy
print(f"Model Accuracy: {accuracy * 100:.2f}%")
