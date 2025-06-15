import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


ads_data = pd.read_csv("advertising.csv")


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


ads_data['Predicted_Sales'] = best_model.predict(scaler.transform(X))


ads_data.to_csv("sales_predictions.csv", index=False)
print("Predictions saved to 'sales_predictions.csv'")
