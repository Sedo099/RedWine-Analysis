# Wine Quality Prediction

This project uses machine learning to predict the quality of wines based on various chemical properties, utilizing the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality). We implemented both Linear Regression and Random Forest models, with hyperparameter tuning using Grid Search.

## Project Structure

- `data/` - Contains the dataset (`winequality.csv`).
- `notebooks/` - Jupyter Notebooks for data analysis and modeling.
- `src/` - Python scripts for prediction and model optimization.
- `README.md` - Project description.

## Libraries Used
The project requires the following libraries:
- `pandas` for data handling
- `matplotlib` and `seaborn` for visualization
- `sklearn` for building and tuning the models

## Project Steps

### 1. Exploratory Data Analysis (EDA)

1. **Initial statistics and visualizations**
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns

   df = pd.read_csv("data/winequality.csv")
   df.describe().round(2)
   sns.histplot(df['quality'], kde=True)
   plt.show()

Correlation Analysis

correlation_matrix = df.corr().round(2)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

Baseline Model - Linear Regression
Linear Regression provides a quick baseline for model comparison:

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = df.drop("quality", axis=1)
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

Random Forest Regression
The Random Forest model captures non-linear relationships and often performs better:

from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(X_train, y_train)
y_pred = forest_model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

4. Hyperparameter Tuning with GridSearchCV
We use GridSearchCV to find the best hyperparameters for the RandomForestRegressor:

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=forest_model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best R-squared Score:", grid_search.best_score_)

Final Model
With optimized parameters, we create a model and evaluate its performance:

best_forest_model = grid_search.best_estimator_
y_pred = best_forest_model.predict(X_test)

print("Optimized Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Optimized R-squared:", r2_score(y_test, y_pred))


This README provides an overview of the project in a GitHub-friendly format, with steps and explanations in English.
