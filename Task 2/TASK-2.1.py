import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("IMDb Movies India.csv", encoding="ISO-8859-1")


df = df[df["Name"].notna() & df["Name"].str.strip().ne("")]
df["Year"] = df["Year"].str.extract(r'(\d{4})')
df["Year"] = pd.to_numeric(df["Year"], errors='coerce')
df["Duration"] = df["Duration"].str.extract(r'(\d+)')
df["Duration"] = pd.to_numeric(df["Duration"], errors='coerce')
df["Votes"] = df["Votes"].str.replace(",", "", regex=False)
df["Votes"] = pd.to_numeric(df["Votes"], errors='coerce')
df = df[df["Rating"].notna()]


for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
    df[col] = df[col].fillna("Unknown")
df["Duration"] = SimpleImputer(strategy="median").fit_transform(df[["Duration"]])


df["Genre"] = df["Genre"].str.split(r",\s*")
mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(df["Genre"]), columns=mlb.classes_, index=df.index)


for col in ["Director", "Actor 1", "Actor 2", "Actor 3"]:
    df[col] = LabelEncoder().fit_transform(df[col])


X = pd.concat([
    df[["Year", "Duration", "Votes", "Director", "Actor 1", "Actor 2", "Actor 3"]],
    genre_encoded
], axis=1)
y = df["Rating"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(" Random Forest RMSE:", round(rmse, 2))


importances = model.feature_importances_
feat_names = X.columns
feat_imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x="Importance", y="Feature", color="skyblue")
plt.title("Top 15 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

predictions_df = X_test.copy()
predictions_df["Actual_Rating"] = y_test.values
predictions_df["Predicted_Rating"] = y_pred
predictions_df.to_csv("rf_imdb_predictions.csv", index=False)

print("\n Predictions saved to 'rf_imdb_predictions.csv'")

