'''
Avocado price prediction using machine learning
Author: Henry Ha
'''
# Import libraries
import pandas as pd

# Load the dataset
df = pd.read_csv('avocado.csv')

#TODO EDA

# Display the first few rows
print(df.head())

# Display general information about the dataset
print(df.info())

# Summary statistics for numerical features
print(df.describe())

# Price trend over time
import matplotlib.pyplot as plt

df['Date'] = pd.to_datetime(df['Date'])
df_grouped = df.groupby('Date').mean()

plt.figure(figsize=(12, 6))
plt.plot(df_grouped.index, df_grouped['AveragePrice'], label='Average Price')
plt.title('Average Avocado Price Over Time')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend()
plt.show()

# Price by type
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(x='type', y='AveragePrice', data=df)
plt.title('Price Distribution by Type')
plt.xlabel('Type')
plt.ylabel('Average Price')
plt.show()

# Sales volume and price correlation
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Total Volume', y='AveragePrice', data=df)
plt.title('Total Volume vs. Average Price')
plt.xlabel('Total Volume')
plt.ylabel('Average Price')
plt.show()

# Regional price analysis
regional_avg_price = df.groupby('region')['AveragePrice'].mean().sort_values()

plt.figure(figsize=(12, 8))
regional_avg_price.plot(kind='bar', color='skyblue')
plt.title('Average Price by Region')
plt.xlabel('Region')
plt.ylabel('Average Price')
plt.xticks(rotation=90)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Scatterplots for linearity check
sns.pairplot(df, y_vars='AveragePrice', x_vars=['Total Volume', '4046', '4225', '4770'])
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()


#TODO Data Preprocessing

# One-hot encoding for the 'type' and 'region' columns
df = pd.get_dummies(df, columns=['type', 'region'], drop_first=True)

# Drop unneeded columns
df = df.drop(columns=['Unnamed: 0'])

# Extract month and day from the 'Date' column
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

X = df.drop(columns=['AveragePrice', 'Date'])
y = df['AveragePrice']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#TODO Model Building

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

# Train the model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

from statsmodels.stats.stattools import durbin_watson

# Calculate residuals
residuals = y_test - lr_model.predict(X_test)

# Durbin-Watson test
print("Durbin-Watson Statistic:", durbin_watson(residuals))

# Homoscedasticity Check
plt.scatter(lr_model.predict(X_test), residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residuals vs. Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()

#TODO Model Evaluation

# Linear Regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predictions
y_pred_lr = lr_model.predict(X_test)

# Metrics
print("Linear Regression:")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R²:", r2_score(y_test, y_pred_lr))

# Visualization of predicted prices vs. actual prices of the linear regression model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.title("Linear Regression: Predictions vs. Actual Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()

# Decision Tree Regressor
# Predictions
y_pred_dt = dt_model.predict(X_test)

# Metrics
print("Decision Tree Regressor:")
print("MAE:", mean_absolute_error(y_test, y_pred_dt))
print("MSE:", mean_squared_error(y_test, y_pred_dt))
print("R²:", r2_score(y_test, y_pred_dt))

# Visualization of predicted prices vs. actual prices of the decision tree regressor model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_dt, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.title("Decision Tree: Predictions vs. Actual Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()

# Random Forest Regressor

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Metrics
print("Random Forest Regressor:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R²:", r2_score(y_test, y_pred_rf))

# Visualization of predicted prices vs. actual prices of the random forest regressor model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.title("Random Forest: Predictions vs. Actual Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()

#TODO Hyperparameter Tuning and Optimization

from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best R² Score:", grid_search.best_score_)

# Train the Random Forest Regressor with the best parameters
best_rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
best_rf_model.fit(X_train, y_train)

# Evaluate on the test set
y_pred_tuned = best_rf_model.predict(X_test)

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Tuned Random Forest Regressor:")
print("MAE:", mean_absolute_error(y_test, y_pred_tuned))
print("MSE:", mean_squared_error(y_test, y_pred_tuned))
print("R²:", r2_score(y_test, y_pred_tuned))


