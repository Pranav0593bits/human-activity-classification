
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Human Activity Classification",
    page_icon="🏃",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏃 Human Activity Classification")

st.write(
    "Classification of human activities using five machine learning "
    "models trained on the UCI Human Activity Recognition Using "
    "Smartphones dataset."
)

st.markdown("---")


# ============================================================
# ACTIVITY MAPPING
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
# MODEL PATHS
# ============================================================

MODEL_DIR = "model"

model_paths = {
    "Logistic Regression": os.path.join(
        MODEL_DIR, "logistic_regression.pkl"
    ),

    "Decision Tree": os.path.join(
        MODEL_DIR, "decision_tree.pkl"
    ),

    "KNN": os.path.join(
        MODEL_DIR, "knn.pkl"
    ),

    "Gaussian Naive Bayes": os.path.join(
        MODEL_DIR, "naive_bayes.pkl"
    ),

    "Random Forest": os.path.join(
        MODEL_DIR, "random_forest.pkl"
    )
}

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    for model_name, path in model_paths.items():

        if os.path.exists(path):
            models[model_name] = joblib.load(path)

    scaler = None

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

    return models, scaler


models, scaler = load_models()


# ============================================================
# CHECK MODEL AVAILABILITY
# ============================================================

if not models:

    st.error(
        "No trained models were found. "
        "Please make sure the model folder is present."
    )

    st.stop()


if scaler is None:

    st.error(
        "Scaler file was not found. "
        "Please make sure model/scaler.pkl exists."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🤖 Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a classification model:",
    list(models.keys())
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Five classification models are available:\n\n"
    "• Logistic Regression\n"
    "• Decision Tree\n"
    "• KNN\n"
    "• Gaussian Naive Bayes\n"
    "• Random Forest"
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📂 Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing test samples.",
    type=["csv"]
)


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    try:

        # --------------------------------------------------------
        # Read CSV
        # --------------------------------------------------------

        data = pd.read_csv(uploaded_file)

        st.success(
            f"File uploaded successfully! "
            f"{data.shape[0]} rows and "
            f"{data.shape[1]} columns detected."
        )


        # --------------------------------------------------------
        # Dataset Preview
        # --------------------------------------------------------

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )


        # --------------------------------------------------------
        # Remove target columns
        # --------------------------------------------------------

        excluded_columns = [
            "Activity",
            "Activity_Name"
        ]

        feature_data = data.drop(
            columns=[
                col
                for col in excluded_columns
                if col in data.columns
            ],
            errors="ignore"
        )


        # --------------------------------------------------------
        # Feature validation
        # --------------------------------------------------------

        expected_features = scaler.n_features_in_

        st.write(
            f"**Features detected:** {feature_data.shape[1]}"
        )

        st.write(
            f"**Features required by scaler:** {expected_features}"
        )


        if feature_data.shape[1] != expected_features:

            st.error(
                f"Feature count mismatch. "
                f"The uploaded CSV contains "
                f"{feature_data.shape[1]} features, "
                f"but the trained scaler expects "
                f"{expected_features} features."
            )

            st.stop()


        # --------------------------------------------------------
        # Convert all features to numeric
        # --------------------------------------------------------

        X_input = feature_data.apply(
            pd.to_numeric,
            errors="coerce"
        )


        # --------------------------------------------------------
        # Check missing/non-numeric values
        # --------------------------------------------------------

        if X_input.isnull().any().any():

            st.error(
                "The uploaded dataset contains missing "
                "or non-numeric values."
            )

            st.stop()


        # --------------------------------------------------------
        # Prediction button
        # --------------------------------------------------------

        if st.button(
            "🔮 Predict Activities",
            type="primary"
        ):

            with st.spinner(
                "Generating predictions..."
            ):

                # ------------------------------------------------
                # IMPORTANT:
                # Convert DataFrame to NumPy before scaling.
                #
                # This avoids sklearn feature-name/order errors
                # while preserving the exact 561-column order
                # present in the uploaded test data.
                # ------------------------------------------------

                X_array = X_input.to_numpy()

                st.write(
                    f"Input shape: {X_array.shape}"
                )


                # ------------------------------------------------
                # Final shape check
                # ------------------------------------------------

                if X_array.shape[1] != expected_features:

                    st.error(
                        f"Final input contains "
                        f"{X_array.shape[1]} features, "
                        f"but the scaler requires "
                        f"{expected_features}."
                    )

                    st.stop()


                # ------------------------------------------------
                # Scale
                # ------------------------------------------------

                X_scaled = scaler.transform(
                    X_array
                )


                # ------------------------------------------------
                # Selected model
                # ------------------------------------------------

                model = models[selected_model]


                # ------------------------------------------------
                # Predictions
                # ------------------------------------------------

                predictions = model.predict(
                    X_scaled
                )


                # ------------------------------------------------
                # Convert predictions to activity names
                # ------------------------------------------------

                prediction_names = [
                    activity_mapping.get(
                        int(pred),
                        str(pred)
                    )
                    for pred in predictions
                ]


            # ====================================================
            # PREDICTION SUCCESS
            # ====================================================

            st.success(
                "Prediction completed successfully!"
            )


            # ====================================================
            # PREDICTION SUMMARY
            # ====================================================

            st.subheader(
                "🎯 Prediction Results"
            )

            prediction_counts = pd.Series(
                prediction_names
            ).value_counts()


            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Samples Classified",
                    len(predictions)
                )

            with col2:

                st.metric(
                    "Selected Model",
                    selected_model
                )


            # ====================================================
            # ACTIVITY DISTRIBUTION
            # ====================================================

            st.subheader(
                "📊 Activity Distribution"
            )

            distribution_df = pd.DataFrame({
                "Activity": prediction_counts.index,
                "Count": prediction_counts.values
            })

            st.bar_chart(
                distribution_df.set_index(
                    "Activity"
                )
            )


            # ====================================================
            # DETAILED PREDICTIONS
            # ====================================================

            st.subheader(
                "📋 Detailed Predictions"
            )

            prediction_output = pd.DataFrame({
                "Predicted Activity": prediction_names
            })


            # ----------------------------------------------------
            # If actual labels exist
            # ----------------------------------------------------

            if "Activity" in data.columns:

                actual_names = [
                    activity_mapping.get(
                        int(value),
                        str(value)
                    )
                    for value in data["Activity"]
                ]


                prediction_output.insert(
                    0,
                    "Actual Activity",
                    actual_names
                )


                prediction_output["Correct"] = (
                    prediction_output[
                        "Actual Activity"
                    ]
                    ==
                    prediction_output[
                        "Predicted Activity"
                    ]
                )


            st.dataframe(
                prediction_output.head(100),
                use_container_width=True
            )


            # ====================================================
            # CONFUSION MATRIX
            # ====================================================

            if "Activity" in data.columns:

                st.markdown("---")

                st.subheader(
                    "📊 Confusion Matrix"
                )


                actual_labels = (
                    data["Activity"]
                    .astype(int)
                    .values
                )

                predicted_labels = (
                    np.array(predictions)
                    .astype(int)
                )


                class_labels = [
                    1, 2, 3, 4, 5, 6
                ]


                cm = confusion_matrix(
                    actual_labels,
                    predicted_labels,
                    labels=class_labels
                )


                activity_names = [
                    "WALKING",
                    "WALKING_UPSTAIRS",
                    "WALKING_DOWNSTAIRS",
                    "SITTING",
                    "STANDING",
                    "LAYING"
                ]


                cm_df = pd.DataFrame(
                    cm,
                    index=activity_names,
                    columns=activity_names
                )


                st.dataframe(
                    cm_df,
                    use_container_width=True
                )


                st.caption(
                    "Rows represent actual activities and "
                    "columns represent predicted activities."
                )


                # =================================================
                # CLASSIFICATION REPORT
                # =================================================

                st.subheader(
                    "📋 Classification Report"
                )


                report = classification_report(
                    actual_labels,
                    predicted_labels,
                    labels=class_labels,
                    target_names=activity_names,
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


            # ====================================================
            # MODEL PERFORMANCE
            # ====================================================

            st.markdown("---")

            st.subheader(
                "📈 Model Performance"
            )


            results_file = "model_results.csv"


            if os.path.exists(results_file):

                results = pd.read_csv(
                    results_file
                )


                model_result = results[
                    results["Model"]
                    == selected_model
                ]


                if not model_result.empty:

                    row = model_result.iloc[0]


                    col1, col2, col3 = st.columns(3)


                    col1.metric(
                        "Accuracy",
                        f"{row['Accuracy']:.2%}"
                    )


                    col2.metric(
                        "AUC",
                        f"{row['AUC']:.2%}"
                    )


                    col3.metric(
                        "Precision",
                        f"{row['Precision']:.2%}"
                    )


                    col4, col5, col6 = st.columns(3)


                    col4.metric(
                        "Recall",
                        f"{row['Recall']:.2%}"
                    )


                    col5.metric(
                        "F1 Score",
                        f"{row['F1 Score']:.2%}"
                    )


                    col6.metric(
                        "MCC",
                        f"{row['MCC']:.2%}"
                    )

                else:

                    st.warning(
                        "Performance results for the selected "
                        "model were not found."
                    )

            else:

                st.warning(
                    "model_results.csv was not found."
                )


            # ====================================================
            # DOWNLOAD PREDICTIONS
            # ====================================================

            st.markdown("---")

            prediction_csv = (
                prediction_output.to_csv(
                    index=False
                )
            )


            st.download_button(
                label="⬇️ Download Predictions",
                data=prediction_csv,
                file_name="activity_predictions.csv",
                mime="text/csv"
            )


    except Exception as e:

        st.error(
            f"An error occurred while processing the file: {e}"
        )


else:

    st.info(
        "Please upload test_data.csv to begin classification."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Human Activity Recognition | "
    "Machine Learning Classification Project"
)
