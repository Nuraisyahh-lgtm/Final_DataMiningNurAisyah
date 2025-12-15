
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

st.set_page_config(page_title="Telco Churn (Ensemble)", layout="wide")
st.title("Telco Customer Churn Prediction")
st.caption("Ensemble Voting: Logistic Regression + RandomForest + GradientBoosting")

# Load artifacts
model = joblib.load("churn_model.joblib")
raw_cols = joblib.load("raw_cols.joblib")
cat_uniques = joblib.load("cat_uniques.joblib")

tabs = st.tabs(["Prediksi", "Visualisasi Hasil Klasifikasi"])

# =========================
# TAB 1: Prediksi
# =========================
with tabs[0]:
    st.subheader("Input Manual")

    with st.sidebar:
        st.header("Input Fitur")
        inputs = {}

        for c in raw_cols:
            if c in cat_uniques:
                options = [""] + cat_uniques[c]
                inputs[c] = st.selectbox(c, options=options, index=0)
            else:
                inputs[c] = st.text_input(c, value="")

        threshold = st.slider("Threshold Churn", 0.1, 0.9, 0.5, 0.05)
        do_pred = st.button("Predict")

    if do_pred:
        X_input = pd.DataFrame([inputs])

        # cast numeric
        for col in X_input.columns:
            if col not in cat_uniques:
                X_input[col] = pd.to_numeric(X_input[col], errors="coerce")

        proba = float(model.predict_proba(X_input)[:, 1][0])
        pred = 1 if proba >= threshold else 0

        st.write("Probabilitas Churn:", proba)
        st.write("Prediksi:", "CHURN (1)" if pred == 1 else "TIDAK CHURN (0)")

        fig = plt.figure()
        plt.bar(["Churn", "Not Churn"], [proba, 1 - proba])
        plt.ylim(0, 1)
        plt.title("Probabilitas Prediksi")
        st.pyplot(fig)

# =========================
# TAB 2: Visualisasi Hasil Klasifikasi
# =========================
with tabs[1]:
    st.subheader("Visualisasi Hasil Klasifikasi (Upload CSV untuk evaluasi)")

    st.write("Upload CSV yang punya kolom target `Churn` (Yes/No atau 1/0).")
    up = st.file_uploader("Upload CSV", type=["csv"])

    if up is None:
        st.info("Belum upload file. Kalau mau lihat confusion matrix + ROC, upload CSV dulu.")
        st.stop()

    df = pd.read_csv(up)

    if "Churn" not in df.columns:
        st.error("CSV harus punya kolom `Churn`.")
        st.stop()

    # normalisasi target
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df = df.dropna().copy()
    y_true = df["Churn"].astype(int)
    X_eval = df.drop(columns=["Churn"])

    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(fig)

    st.metric("ROC AUC", f"{auc:.4f}")
