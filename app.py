import streamlit as st
import pandas as pd

from model import (
    load_data,
    prepare_data,
    split_data,
    train_model,
    evaluate_model,
    feature_importance
)

from utils import (
    plot_confusion_matrix,
    plot_feature_importance
)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Diabetes AI",
    page_icon="🩺",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("🩺 Diabetes Prediction AI")
st.caption("Random Forest Model for Medical Prediction")

# ------------------ LOAD DATA ------------------
df = load_data("data/diabetes.csv")

# ------------------ PREPARE MODEL ------------------
X, y = prepare_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)
model = train_model(X_train, y_train)
acc, cm, report = evaluate_model(model, X_test, y_test)

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("🧾 Patient Input")

user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.slider(
        col,
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )

input_df = pd.DataFrame([user_input])

# ------------------ PREDICTION ------------------
prediction = model.predict(input_df)[0]

# ------------------ MAIN LAYOUT ------------------
col1, col2 = st.columns([1, 2])

# LEFT PANEL
with col1:
    st.subheader("📊 Model Performance")

    st.metric("Accuracy", f"{acc:.2%}")

    st.markdown("### 🧠 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk")

# RIGHT PANEL
with col2:
    st.subheader("📈 Visual Insights")

    tab1, tab2 = st.tabs(["Confusion Matrix", "Feature Importance"])

    with tab1:
        fig_cm = plot_confusion_matrix(cm)
        st.pyplot(fig_cm)

    with tab2:
        importance = feature_importance(model, X)
        fig_fi = plot_feature_importance(importance)
        st.pyplot(fig_fi)

# ------------------ DATA PREVIEW ------------------
with st.expander("🔍 View Dataset"):
    st.dataframe(df.head())