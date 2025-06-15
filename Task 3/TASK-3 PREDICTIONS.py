import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("IRIS.csv")


label_encoder = LabelEncoder()
df["species_encoded"] = label_encoder.fit_transform(df["species"])


X = df.drop(["species", "species_encoded"], axis=1)
y = df["species_encoded"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(" Accuracy:", round(accuracy * 100, 2), "%")


predicted_species = label_encoder.inverse_transform(y_pred)


results_df = X_test.copy()
results_df["Actual_Species"] = label_encoder.inverse_transform(y_test)
results_df["Predicted_Species"] = predicted_species
results_df.to_csv("iris_predictions.csv", index=False)
print(" Predictions saved to 'iris_predictions.csv'")
