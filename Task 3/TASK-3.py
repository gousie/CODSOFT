import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif


df_iris = pd.read_csv('IRIS.csv')


print("Dataset Info:")
print(df_iris.info())
print("\nClass Counts:")
print(df_iris['species'].value_counts())


sns.pairplot(df_iris, hue='species', diag_kind='hist', palette='Set2')
plt.suptitle("Pairwise Feature Relationships by Species", y=1.02)
plt.show()


corr_matrix = df_iris.drop('species', axis=1).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Feature Correlation Matrix")
plt.show()


features = df_iris.drop('species', axis=1)
target = df_iris['species']


encoder = LabelEncoder()
target_encoded = encoder.fit_transform(target)


selector = SelectKBest(score_func=f_classif, k=3)
features_selected = selector.fit_transform(features, target_encoded)
selected_features = features.columns[selector.get_support()].tolist()
print("\nSelected Features:", selected_features)


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_selected)


X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target_encoded, test_size=0.25, random_state=456
)


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=456)

cv_scores = cross_val_score(rf_classifier, features_scaled, target_encoded, cv=5)
print("\nCross-Validation Scores:", cv_scores)
print("Mean CV Accuracy: {:.3f} (±{:.3f})".format(cv_scores.mean(), cv_scores.std() * 2))


rf_classifier.fit(X_train, y_train)


y_pred = rf_classifier.predict(X_test)

accuracy = (y_pred == y_test).mean()
print("\nTest Set Accuracy: {:.3f}".format(accuracy))


print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


importance = pd.Series(rf_classifier.feature_importances_, index=selected_features)
plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind='barh', color='teal')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.show()

actual_species = encoder.inverse_transform(y_test)
predicted_species = encoder.inverse_transform(y_pred)


results_df = pd.DataFrame(X_test, columns=selected_features)
results_df["Actual_Species"] = actual_species
results_df["Predicted_Species"] = predicted_species


results_df.to_csv("iris_rf_predictions.csv", index=False)
print("\n Predictions saved to 'iris_rf_predictions.csv'")

