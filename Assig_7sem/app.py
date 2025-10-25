import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="wide",
    page_icon="🩺"
)

st.title("🩺 Diabetes Risk Analysis App")
st.write("This app predicts diabetes likelihood.")

@st.cache_resource
def load_train_model():
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
    grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
    grid_knn.fit(X_train, y_train)
    best_k = grid_knn.best_params_['n_neighbors']
    knn = KNeighborsClassifier(n_neighbors=best_k)

    ensemble = VotingClassifier(estimators=[('lgbm', lgbm), ('knn', knn)], voting='soft')
    ensemble.fit(X_train, y_train)

    return df, X, y, X_train, X_test, y_train, y_test, scaler, ensemble, lgbm


df, X, y, X_train, X_test, y_train, y_test, scaler, ensemble, lgbm = load_train_model()

st.header("Input Patient Data")

input_data = {}
for feature in X.columns:
    val = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    input_data[feature] = val

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

if st.button("Predict Diabetes Risk"):
    pred_prob = ensemble.predict_proba(input_scaled)[0][1]
    pred_class = "Diabetic" if pred_prob > 0.5 else "Non-Diabetic"

    st.subheader("Prediction Result")
    st.markdown(f"**Prediction:** {pred_class}")
    st.markdown(f"**Probability:** {pred_prob:.2f}")

st.header("Model Explainability (SHAP)")

with st.spinner("Computing SHAP values..."):
    lgbm.fit(X_train, y_train)
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X_train)

    st.subheader("SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, feature_names=X.columns, show=False)
    st.pyplot(fig)
