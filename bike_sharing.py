import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

# API Key
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
!kaggle datasets download -d marklvl/bike-sharing-dataset
!unzip bike-sharing-dataset.zip

current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

files = os.listdir(current_directory)
print("Files:", files)

df = pd.read_csv('/content/bike-sharing-dataset/day.csv')
df.head()

df.describe()

# Data Exploration
print(df.isnull().sum())

# Feature Engineering
df['dteday'] = pd.to_datetime(df['dteday'])

# Encoding categorical features
categorical_columns = ['season', 'holiday', 'workingday', 'weathersit']
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Convert 'dteday' to month, year, and day_of_week
df['month'] = df['dteday'].dt.month.astype('category')
df['year'] = df['dteday'].dt.year.astype('category')
df['day_of_week'] = df['dteday'].dt.dayofweek.astype('category')

categorical_columns = ['season', 'holiday', 'workingday', 'weathersit', 'month', 'year', 'day_of_week']
df = pd.get_dummies(df, columns=categorical_columns)
df['temp_hum_interaction'] = df['temp'] * df['hum']

df['temp_hum_interaction'] = df['temp'] * df['hum']

# Polynomial Features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = poly.fit_transform(df[['temp', 'atemp', 'hum', 'windspeed']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['temp', 'atemp', 'hum', 'windspeed']))

# Combine with original dataframe
df = pd.concat([df, poly_df], axis=1)

print(df.head())

# Checking for missing values
print(df.isnull().sum())

# Filling missing values if any
df.fillna(df.mean(), inplace=True)

# Adding lag features
df['lag_1'] = df['cnt'].shift(1)
df['lag_2'] = df['cnt'].shift(2)
df.bfill(inplace=True)  # Deprecated fillna method replaced with bfill

X = df.drop(['cnt', 'dteday', 'casual', 'registered'], axis=1)
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(df.columns)
print(df.head())

# Preparing the data
df = pd.read_csv('/content/bike-sharing-dataset/day.csv')
df['dteday'] = pd.to_datetime(df['dteday'])
df['day_of_week'] = df['dteday'].dt.dayofweek

# Plot for time series analysis
plt.figure(figsize=(15, 5))
df['cnt'].plot() 
plt.title('Hourly Bike Rentals')
plt.xlabel('Time')
plt.ylabel('Total Count')
plt.show()

# Rental counts by registered and casual users (using histogram)
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='registered', bins=30, kde=True)
plt.title('Registered Users')
plt.xlabel('User Type')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='casual', bins=30, kde=True)
plt.title('Casual Users')
plt.xlabel('User Type')
plt.ylabel('Count')
plt.show()

# Weathersit Mapping
weathersit_mapping = {
    1: 'Clear/Partly Cloudy',
    2: 'Mist/Cloudy',
    3: 'Light Snow/Rain'
}
df['weathersit'] = df['weathersit'].map(weathersit_mapping)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x=df['weathersit'].value_counts().index, y=df['weathersit'].value_counts().values)
plt.title('Bike Rentals by Weather Situation')
plt.xlabel('Weather Situation')
plt.ylabel('Count')
plt.show()

# Day of Week Mapping
day_of_week_mapping = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}
df['day_of_week'] = df['day_of_week'].map(day_of_week_mapping)

# Calculating Aaerage rental numbers per the day of week
day_of_week_avg = df.groupby('day_of_week')['cnt'].mean().reindex(day_of_week_mapping.values())

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x=day_of_week_avg.index, y=day_of_week_avg.values)
plt.title('Average Bike Rentals by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Count')
plt.ylim(0, max(day_of_week_avg.values) * 1.1)  # To expand the Y axis
plt.show()

# Calculating average rental numbers per month
df['month'] = df['dteday'].dt.month
month_avg = df.groupby('month')['cnt'].mean()

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x=month_avg.index, y=month_avg.values)
plt.title('Average Bike Rentals by Month')
plt.xlabel('Month')
plt.ylabel('Average Count')
plt.ylim(0, max(month_avg.values) * 1.1)  # To expand the Y axis
plt.show()

# Data Preparation
df = pd.read_csv('/content/bike-sharing-dataset/day.csv')
df['dteday'] = pd.to_datetime(df['dteday'])
df['month'] = df['dteday'].dt.month.astype('category')
df['year'] = df['dteday'].dt.year.astype('category')
df['day_of_week'] = df['dteday'].dt.dayofweek.astype('category')
categorical_columns = ['season', 'holiday', 'workingday', 'weathersit', 'month', 'year', 'day_of_week']
df = pd.get_dummies(df, columns=categorical_columns)
df['temp_hum_interaction'] = df['temp'] * df['hum']

# Features and target variable
month_columns = [col for col in df.columns if 'month_' in col]
day_of_week_columns = [col for col in df.columns if 'day_of_week_' in col]
feature_columns = ['temp', 'atemp', 'hum', 'windspeed', 'temp_hum_interaction'] + month_columns + day_of_week_columns
X = df[feature_columns]
y = df['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{model_name} Mean Squared Error: {mse}')
    print(f'{model_name} R^2 Score: {r2}')
    
    # Scatter plot for predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title(f'Actual vs Predictions ({model_name})')
    plt.show()

    # Residuals distribution
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50)
    plt.title(f'Residual Distribution ({model_name})')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
evaluate_model(lr_model, X_test, y_test, 'Linear Regression')

# Random Forest Model
rf_model = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None]
}
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='r2')
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
evaluate_model(best_rf_model, X_test, y_test, 'Random Forest')
print(f'Best Parameters for Random Forest: {grid_search_rf.best_params_}')

# Gradient Boosting Model
gb_model = GradientBoostingRegressor()
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=param_grid_gb, cv=5, scoring='r2')
grid_search_gb.fit(X_train, y_train)
best_gb_model = grid_search_gb.best_estimator_
evaluate_model(best_gb_model, X_test, y_test, 'Gradient Boosting')
print(f'Best Parameters for Gradient Boosting: {grid_search_gb.best_params_}')

# Decision Tree Model
dt_model = DecisionTreeRegressor(random_state=42)
param_grid_dt = {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
random_search_dt = RandomizedSearchCV(estimator=dt_model, param_distributions=param_grid_dt, n_iter=10, cv=3, scoring='r2', random_state=42)
random_search_dt.fit(X_train, y_train)
best_dt_model = random_search_dt.best_estimator_
evaluate_model(best_dt_model, X_test, y_test, 'Decision Tree')
print(f'Best Parameters for Decision Tree: {random_search_dt.best_params_}')

# Cross-validation scores for the best models
def cross_val_scores(model, X, y, model_name):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f'Cross-Validation R^2 Scores for {model_name}: {cv_scores}')
    print(f'Average Cross-Validation R^2 Score for {model_name}: {np.mean(cv_scores)}')

cross_val_scores(best_rf_model, X, y, 'Random Forest')
cross_val_scores(best_gb_model, X, y, 'Gradient Boosting')
cross_val_scores(best_dt_model, X, y, 'Decision Tree')

# Feature Scaling and Re-evaluation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Re-train and evaluate models with scaled data
lr_model.fit(X_train_scaled, y_train)
evaluate_model(lr_model, X_test_scaled, y_test, 'Scaled Linear Regression')

dt_model_scaled = DecisionTreeRegressor(random_state=42)
dt_model_scaled.fit(X_train_scaled, y_train)
evaluate_model(dt_model_scaled, X_test_scaled, y_test, 'Scaled Decision Tree')

