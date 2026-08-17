
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Human Activity Classification",
    page_icon="📱",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


# ============================================================
# TITLE
# ============================================================

st.title("📱 Human Activity Classification")

st.markdown(
    """
    ### UCI Human Activity Recognition Using Smartphones

    This interactive application demonstrates five machine learning
    classification models trained on the UCI Human Activity Recognition
    dataset.
    """
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    models["Logistic Regression"] = joblib.load(
        MODEL_DIR / "logistic_regression.pkl"
    )

    models["Decision Tree"] = joblib.load(
        MODEL_DIR / "decision_tree.pkl"
    )

    models["K-Nearest Neighbor"] = joblib.load(
        MODEL_DIR / "knn.pkl"
    )

    models["Gaussian Naive Bayes"] = joblib.load(
        MODEL_DIR / "naive_bayes.pkl"
    )

    models["Random Forest"] = joblib.load(
        MODEL_DIR / "random_forest.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "scaler.pkl"
    )

    return models, scaler


models, scaler = load_models()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Model Controls")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data CSV",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.write("**Dataset:** UCI HAR")
st.sidebar.write("**Instances:** 2,947")
st.sidebar.write("**Features:** 561")
st.sidebar.write("**Classes:** 6")


# ============================================================
# REQUIRE FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload the `test_data.csv` file to begin."
    )

    st.markdown(
        """
        ### Expected File

        Upload the `test_data.csv` supplied with this project.

        The file contains:

        - 561 feature columns
        - Activity
        - Activity_Name
        """
    )

    st.stop()


# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(uploaded_file)


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

non_feature_columns = [
    "Activity",
    "Activity_Name"
]

feature_columns = [
    column
    for column in df.columns
    if column not in non_feature_columns
]


# ============================================================
# VALIDATION
# ============================================================

if len(feature_columns) != 561:

    st.error(
        f"Expected 561 feature columns, but found "
        f"{len(feature_columns)}."
    )

    st.stop()


if "Activity" not in df.columns:

    st.error(
        "The uploaded CSV must contain an 'Activity' column."
    )

    st.stop()


# ============================================================
# DATA INFORMATION
# ============================================================

st.header("📊 Dataset Information")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Test Instances",
    len(df)
)

col2.metric(
    "Features",
    len(feature_columns)
)

col3.metric(
    "Classes",
    df["Activity"].nunique()
)

col4.metric(
    "Selected Model",
    selected_model
)


# ============================================================
# SHOW DATA
# ============================================================

with st.expander("View Test Dataset"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# PREPARE DATA
# ============================================================

X = df[feature_columns]

y_true = df["Activity"]


# ============================================================
# SCALE DATA
# ============================================================

X_scaled = scaler.transform(X)


# ============================================================
# SELECT MODEL
# ============================================================

model = models[selected_model]


# KNN, Logistic Regression and Naive Bayes
# use scaled features.

if selected_model in [
    "Logistic Regression",
    "K-Nearest Neighbor",
    "Gaussian Naive Bayes"
]:

    X_input = X_scaled

else:

    X_input = X


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_input
)

y_prob = model.predict_proba(
    X_input
)


# ============================================================
# ACTIVITY LABELS
# ============================================================

activity_mapping = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}


# ============================================================
# MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

auc = roc_auc_score(
    y_true,
    y_prob,
    multi_class="ovr",
    average="weighted"
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

mcc = matthews_corrcoef(
    y_true,
    y_pred
)


# ============================================================
# DISPLAY METRICS
# ============================================================

st.header(
    f"📈 Evaluation Metrics — {selected_model}"
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

c2.metric(
    "AUC Score",
    f"{auc:.4f}"
)

c3.metric(
    "Precision",
    f"{precision:.4f}"
)


c4, c5, c6 = st.columns(3)

c4.metric(
    "Recall",
    f"{recall:.4f}"
)

c5.metric(
    "F1 Score",
    f"{f1:.4f}"
)

c6.metric(
    "MCC Score",
    f"{mcc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.header("🔢 Confusion Matrix")

labels = sorted(
    activity_mapping.keys()
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[
        activity_mapping[x]
        for x in labels
    ],
    columns=[
        activity_mapping[x]
        for x in labels
    ]
)

st.dataframe(
    cm_df,
    use_container_width=True
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.header("📋 Classification Report")

report = classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=[
        activity_mapping[x]
        for x in labels
    ],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

st.header("🔮 Sample Predictions")

prediction_df = pd.DataFrame({
    "Actual Activity ID": y_true.values,
    "Actual Activity": [
        activity_mapping[int(x)]
        for x in y_true
    ],
    "Predicted Activity ID": y_pred,
    "Predicted Activity": [
        activity_mapping[int(x)]
        for x in y_pred
    ]
})

st.dataframe(
    prediction_df.head(50),
    use_container_width=True
)


# ============================================================
# ALL MODEL COMPARISON
# ============================================================

st.header("🏆 Comparison of All Classification Models")

comparison_results = []

for model_name, current_model in models.items():

    if model_name in [
        "Logistic Regression",
        "K-Nearest Neighbor",
        "Gaussian Naive Bayes"
    ]:

        current_X = X_scaled

    else:

        current_X = X

    current_pred = current_model.predict(
        current_X
    )

    current_prob = current_model.predict_proba(
        current_X
    )

    comparison_results.append({

        "ML Model Name":
            model_name,

        "Accuracy":
            accuracy_score(
                y_true,
                current_pred
            ),

        "AUC":
            roc_auc_score(
                y_true,
                current_prob,
                multi_class="ovr",
                average="weighted"
            ),

        "Precision":
            precision_score(
                y_true,
                current_pred,
                average="weighted",
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                current_pred,
                average="weighted",
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                current_pred,
                average="weighted",
                zero_division=0
            ),

        "MCC":
            matthews_corrcoef(
                y_true,
                current_pred
            )
    })


comparison_df = pd.DataFrame(
    comparison_results
)


st.dataframe(
    comparison_df.round(4),
    use_container_width=True
)


# ============================================================
# BEST MODEL
# ============================================================

best_model_row = comparison_df.loc[
    comparison_df["F1"].idxmax()
]

st.success(
    f"🏆 Best model based on weighted F1 Score: "
    f"**{best_model_row['ML Model Name']}**"
)


# ============================================================
# PERFORMANCE CHART
# ============================================================

st.header("📊 Model Performance Comparison")

chart_df = comparison_df.set_index(
    "ML Model Name"
)[
    [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
]

st.bar_chart(
    chart_df
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Machine Learning Classification Assignment — "
    "UCI Human Activity Recognition Using Smartphones Dataset"
)
