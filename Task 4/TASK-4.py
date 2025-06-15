import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


ads_data = pd.read_csv('advertising.csv')


print("Dataset Summary:")
print(ads_data.describe())
print("\nMissing Values:")
print(ads_data.isnull().sum())  


plt.figure(figsize=(8, 6))
sns.heatmap(ads_data.corr(), annot=True, cmap='viridis', vmin=-1, vmax=1)
plt.title("Correlation Matrix of Advertising Features and Sales")
plt.show()


plt.figure(figsize=(12, 4))
for i, col in enumerate(['TV', 'Radio', 'Newspaper']):
    plt.subplot(1, 3, i+1)
    sns.histplot(ads_data[col], kde=True, color='purple')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 4))
for i, col in enumerate(['TV', 'Radio', 'Newspaper']):
    plt.subplot(1, 3, i+1)
    plt.scatter(ads_data[col], ads_data['Sales'], alpha=0.5, color='teal')
    plt.xlabel(col)
    plt.ylabel('Sales')
    plt.title(f'{col} vs. Sales')
plt.tight_layout()
plt.show()


ads_data['TV_Radio'] = ads_data['TV'] * ads_data['Radio']
ads_data['TV_Newspaper'] = ads_data['TV'] * ads_data['Newspaper']
ads_data['Radio_Newspaper'] = ads_data['Radio'] * ads_data['Newspaper']


X = ads_data[['TV', 'Radio', 'Newspaper', 'TV_Radio', 'TV_Newspaper', 'Radio_Newspaper']]
y = ads_data['Sales']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=789)


gb_model = GradientBoostingRegressor(random_state=789)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
print("\nBest Hyperparameters:", grid_search.best_params_)


y_pred = best_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("\nModel Performance on Test Set:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error: {mae:.2f}")


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='darkblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.show()


feature_names = X.columns
importance = pd.Series(best_model.feature_importances_, index=feature_names)
plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind='barh', color='darkgreen')
plt.title('Feature Importance (Gradient Boosting)')
plt.xlabel('Importance')
plt.show()


print("\nBusiness Recommendations:")
print("- Allocate more budget to TV and Radio advertising, as they have the highest impact on sales.")
print("- Consider interaction effects (e.g., TV and Radio combined) for campaign planning.")
print("- Newspaper advertising has lower impact; evaluate its cost-effectiveness.")
