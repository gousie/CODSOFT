import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
data = pd.read_csv('Titanic-Dataset.csv')
passenger_info = data[['PassengerId', 'Name']].copy()


data = data.drop(['Ticket', 'Cabin'], axis=1)


data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])


data['FamilySize'] = data['SibSp'] + data['Parch'] + 1


X = data.drop(['Survived', 'PassengerId', 'Name'], axis=1)
y = data['Survived']


numeric_features = ['Age', 'Fare', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=4, random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)


y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {accuracy:.4f}")


y_pred_all = model.predict(X)


output = passenger_info.copy()
output['Predicted_Survived'] = y_pred_all
output['Predicted_Survived'] = output['Predicted_Survived'].map({0: 'Did Not Survive', 1: 'Survived'})


print("\nPredictions for Each Passenger:")
print(output.head(10))


output.to_csv('titanic_predictions.csv', index=False)
print("\nPredictions saved to 'titanic_predictions.csv'")


feature_names = numeric_features + [
    f"{feature}_{category}" 
    for feature, categories in zip(categorical_features, 
        model.named_steps['preprocessor'].named_transformers_['cat'].categories_)
    for category in categories[1:]
]

plt.figure(figsize=(20, 10))
plot_tree(
    model.named_steps['classifier'],
    feature_names=feature_names,
    class_names=['Did Not Survive', 'Survived'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Titanic Survival Prediction")
plt.savefig('titanic_decision_tree.png')
plt.show()
print("\nDecision tree visualization saved as 'titanic_decision_tree.png'")
