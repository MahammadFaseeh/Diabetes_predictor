import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
import lightgbm as lgb

df = pd.read_csv(r"D:\Assig_7sem\DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv")

if df['gender'].dtype == 'object':
    df['gender'] = LabelEncoder().fit_transform(df['gender'])

df['diabetic'] = df['diabetic'].map({'Yes': 1, 'No': 0})

df = df.dropna()

X = df.drop(columns=['diabetic'])
y = df['diabetic']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

lgbm = lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metric='auc',
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42
)

param_grid_knn = {'n_neighbors': [5, 11, 15, 21, 23, 25]}
knn = KNeighborsClassifier()
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy')
grid_knn.fit(X_train, y_train)
best_k = grid_knn.best_params_['n_neighbors']
print(f"Best K found for KNN: {best_k}")
knn = KNeighborsClassifier(n_neighbors=best_k)

ensemble = VotingClassifier(
    estimators=[('lgbm', lgbm), ('knn', knn)],
    voting='soft'
)

ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
y_prob = ensemble.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(ensemble, X_test, y_test)
plt.show()

lgbm.fit(X_train, y_train)
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, feature_names=df.drop(columns=['diabetic']).columns)



