import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
df = pd.read_csv("creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Fast Random Forest Performance:\n")
print(classification_report(y_test, y_pred))
results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred
results.to_csv("fast_fraud_predictions.csv", index=False)

print("\n Saved to 'fast_fraud_predictions.csv'")
input("Press Enter to exit...")
