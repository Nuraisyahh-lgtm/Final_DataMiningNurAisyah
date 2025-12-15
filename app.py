import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
)
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)

# ======================
# CONFIG
# ======================
RANDOM_STATE = 42
MODEL_DIR = "models"
DATA_DIR = "data"

MODEL_CLF_PATH = os.path.join(MODEL_DIR, "classifier_voting.joblib")
MODEL_REG_PATH = os.path.join(MODEL_DIR, "regressor_voting.joblib")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.json")
DATA_PATH = os.path.join(DATA_DIR, "wine_quality.csv")

RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

st.set_page_config(page_title="Wine Quality - Classification + Regression", layout="wide")
st.title("Wine Quality - Ensemble Classification + Regression")


# ======================
# TRAIN / PREPARE FILES (untuk Opsi 1: hanya upload app.py + requirements.txt)
# ======================
def train_if_needed() -> pd.DataFrame:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) Pastikan dataset ada
    if not os.path.exists(DATA_PATH):
        red = pd.read_csv(RED_URL, sep=";")
        white = pd.read_csv(WHITE_URL, sep=";")

        red["wine_type"] = 0
        white["wine_type"] = 1

        df = pd.concat([red, white], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    # 2) Pastikan feature cols tersimpan
    feature_cols = list(df.drop(columns=["quality"]).columns)
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(feature_cols, f)

    # 3) Kalau model belum ada, train dan simpan
    if not (os.path.exists(MODEL_CLF_PATH) and os.path.exists(MODEL_REG_PATH)):
        X = df.drop(columns=["quality"]).copy()
        y_clf = (df["quality"] >= 7).astype(int)  # klasifikasi: good jika >= 7
        y_reg = df["quality"].astype(float)       # regresi: prediksi quality asli

        X_train, _, y_train_clf, _, y_train_reg, _ = train_test_split(
            X, y_clf, y_reg,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_clf
        )

        # Ensemble Classification (selain KNN & DT)
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        ])
        rf = RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1)
        gb = GradientBoostingClassifier(random_state=RANDOM_STATE)

        clf = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
            voting="soft"
        )
        clf.fit(X_train, y_train_clf)

        # Ensemble Regression
        rf_reg = RandomForestRegressor(n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1)
        gbr_reg = GradientBoostingRegressor(random_state=RANDOM_STATE)
        et_reg = ExtraTreesRegressor(n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1)

        reg = VotingRegressor(
            estimators=[("rf", rf_reg), ("gbr", gbr_reg), ("et", et_reg)]
        )
        reg.fit(X_train, y_train_reg)

        joblib.dump(clf, MODEL_CLF_PATH, compress=3)
        joblib.dump(reg, MODEL_REG_PATH, compress=3)

    return df


# ======================
# CACHING
# ======================
@st.cache_data
def load_data():
    return train_if_needed()

@st.cache_resource
def load_models():
    clf = joblib.load(MODEL_CLF_PATH)
    reg = joblib.load(MODEL_REG_PATH)
    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)
    return clf, reg, feature_cols


df = load_data()
clf, reg, feature_cols = load_models()


# ======================
# DATA PREP
# ======================
X = df.drop(columns=["quality"]).copy()
y_reg = df["quality"].astype(float)
y_clf = (df["quality"] >= 7).astype(int)

X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = train_test_split(
    X, y_clf, y_reg,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_clf
)


# ======================
# UI
# ======================
menu = st.sidebar.radio("Menu", ["Dataset", "Klasifikasi", "Regresi", "Evaluasi"])

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
    ConfusionMatrixDisplay(cm, display_labels=["Not Good(0)", "Good(1)"]).plot(ax=ax, values_format="d")
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
                user_in[col] = st.slider(col, min_value=vmin, max_value=vmax, value=vdefault, key="reg_" + col)
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
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_pred_reg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy (clf)", f"{acc:.3f}")
    c2.metric("MAE (reg)", f"{mae:.3f}")
    c3.metric("RMSE (reg)", f"{rmse:.3f}")
    c4.metric("R2 (reg)", f"{r2:.3f}")

    st.caption("Evaluasi ditampilkan untuk bukti performa model pada laporan/presentasi.")
