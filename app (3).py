
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

st.set_page_config(page_title="Wine Quality - Classification + Regression", layout="wide")
st.title("Wine Quality - Ensemble Classification + Regression")

@st.cache_data
def load_data():
    df = pd.read_csv("data/wine_quality.csv")
    return df

@st.cache_resource
def load_models():
    clf = joblib.load("models/classifier_voting.joblib")
    reg = joblib.load("models/regressor_voting.joblib")
    with open("models/feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    return clf, reg, feature_cols

df = load_data()
clf, reg, feature_cols = load_models()

# Sidebar menu
menu = st.sidebar.radio("Menu", ["Dataset", "Klasifikasi", "Regresi", "Evaluasi"])

# Prepare X, y
X = df.drop(columns=["quality"]).copy()
y_reg = df["quality"].astype(float)
y_clf = (df["quality"] >= 7).astype(int)

# split fixed biar hasil evaluasi konsisten
X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = train_test_split(
    X, y_clf, y_reg,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_clf
)

if menu == "Dataset":
    st.subheader("Preview Dataset")
    st.write("Sumber: UCI Wine Quality (red + white), digabung, tambah fitur wine_type (0=red, 1=white).")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Statistik Singkat")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah data", len(df))
    c2.metric("Jumlah fitur", len(feature_cols))
    c3.metric("Good (quality >= 7)", int((df["quality"] >= 7).sum()))

elif menu == "Klasifikasi":
    st.subheader("Prediksi Klasifikasi (Good vs Not Good) - Ensemble VotingClassifier")

    st.caption("Label: Good = quality >= 7")

    # input form
    with st.form("form_clf"):
        cols = st.columns(3)
        user_in = {}
        for i, col in enumerate(feature_cols):
            with cols[i % 3]:
                vmin = float(df[col].min())
                vmax = float(df[col].max())
                vdefault = float(df[col].median())
                user_in[col] = st.slider(col, min_value=vmin, max_value=vmax, value=vdefault)
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        X_user = pd.DataFrame([user_in])[feature_cols]
        pred = int(clf.predict(X_user)[0])
        proba = float(clf.predict_proba(X_user)[0, 1])

        label = "Good (1)" if pred == 1 else "Not Good (0)"
        st.success(f"Hasil prediksi: {label}")
        st.info(f"Probabilitas Good: {proba:.3f}")

    st.divider()
    st.subheader("Visualisasi Confusion Matrix (Test Set)")
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test_clf, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Good(0)", "Good(1)"])
    disp.plot(ax=ax, values_format="d")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    st.subheader("Visualisasi ROC Curve (Test Set)")
    y_proba = clf.predict_proba(X_test)[:, 1]
    fig2, ax2 = plt.subplots()
    RocCurveDisplay.from_predictions(y_test_clf, y_proba, ax=ax2)
    plt.title("ROC Curve")
    st.pyplot(fig2)

elif menu == "Regresi":
    st.subheader("Prediksi Regresi (Quality 0-10) - Ensemble VotingRegressor")

    with st.form("form_reg"):
        cols = st.columns(3)
        user_in = {}
        for i, col in enumerate(feature_cols):
            with cols[i % 3]:
                vmin = float(df[col].min())
                vmax = float(df[col].max())
                vdefault = float(df[col].median())
                user_in[col] = st.slider(col, min_value=vmin, max_value=vmax, value=vdefault, key="reg_"+col)
        submitted = st.form_submit_button("Prediksi Quality")

    if submitted:
        X_user = pd.DataFrame([user_in])[feature_cols]
        pred_q = float(reg.predict(X_user)[0])
        st.success(f"Prediksi quality: {pred_q:.2f}")

    st.divider()
    st.subheader("Visualisasi Actual vs Predicted (Test Set)")
    y_pred_reg = reg.predict(X_test)

    fig, ax = plt.subplots()
    ax.scatter(y_test_reg, y_pred_reg, alpha=0.5)
    ax.set_xlabel("Actual Quality")
    ax.set_ylabel("Predicted Quality")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

elif menu == "Evaluasi":
    st.subheader("Ringkasan Evaluasi Model")

    # klasifikasi
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    acc = (y_pred == y_test_clf).mean()

    # regresi
    y_pred_reg = reg.predict(X_test)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
    r2 = r2_score(y_test_reg, y_pred_reg)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy (clf)", f"{acc:.3f}")
    c2.metric("MAE (reg)", f"{mae:.3f}")
    c3.metric("R2 (reg)", f"{r2:.3f}")

    st.caption("Tujuan evaluasi: bukti performa model untuk laporan/presentasi.")
