
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Human Activity Classification",
    page_icon="🏃",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏃 Human Activity Classification")

st.write(
    "Classification of human activities using five machine learning "
    "models trained on the UCI Human Activity Recognition Using "
    "Smartphones dataset."
)

st.markdown("---")

# --------------------------------------------------
# Activity mapping
# --------------------------------------------------

activity_mapping = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# --------------------------------------------------
# Model paths
# --------------------------------------------------

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

# --------------------------------------------------
# Load models
# --------------------------------------------------

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

# --------------------------------------------------
# Check model availability
# --------------------------------------------------

if not models:

    st.error(
        "No trained models were found. "
        "Please make sure the model folder is present."
    )

    st.stop()

if scaler is None:

    st.error(
        "Scaler was not found. "
        "Please make sure model/scaler.pkl exists."
    )

    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a classification model:",
    list(models.keys())
)

st.sidebar.markdown("---")

st.sidebar.info(
    "The application supports five classification models: "
    "Logistic Regression, Decision Tree, KNN, "
    "Gaussian Naive Bayes, and Random Forest."
)

# --------------------------------------------------
# File upload
# --------------------------------------------------

st.header("📂 Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing test samples.",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # --------------------------------------------------
        # Read CSV
        # --------------------------------------------------

        data = pd.read_csv(uploaded_file)

        st.success(
            f"File uploaded successfully! "
            f"{data.shape[0]} rows and "
            f"{data.shape[1]} columns detected."
        )

        # --------------------------------------------------
        # Dataset preview
        # --------------------------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

        # --------------------------------------------------
        # Remove target columns
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Validate number of features
        # --------------------------------------------------

        if feature_data.shape[1] != 561:

            st.error(
                f"The uploaded file contains "
                f"{feature_data.shape[1]} feature columns. "
                f"The model requires exactly 561 features."
            )

            st.stop()

        # --------------------------------------------------
        # Prediction button
        # --------------------------------------------------

        if st.button(
            "🔮 Predict Activities",
            type="primary"
        ):

            with st.spinner(
                "Generating predictions..."
            ):

                # --------------------------------------------------
                # Convert values to numeric
                # --------------------------------------------------

                X_input = feature_data.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                # --------------------------------------------------
                # Check missing values
                # --------------------------------------------------

                if X_input.isnull().any().any():

                    st.error(
                        "The uploaded dataset contains "
                        "missing or non-numeric values."
                    )

                    st.stop()

                # --------------------------------------------------
                # Get scaler feature names
                # --------------------------------------------------

                if hasattr(
                    scaler,
                    "feature_names_in_"
                ):

                    expected_features = list(
                        scaler.feature_names_in_
                    )

                    # --------------------------------------------------
                    # Restore duplicate column names
                    # --------------------------------------------------

                    corrected_columns = []

                    for col in X_input.columns:

                        corrected_col = col

                        if "." in col:

                            base_name, suffix = col.rsplit(
                                ".",
                                1
                            )

                            if suffix.isdigit():

                                corrected_col = base_name

                        corrected_columns.append(
                            corrected_col
                        )

                    X_input.columns = corrected_columns

                    # --------------------------------------------------
                    # Create occurrence numbers
                    # --------------------------------------------------

                    csv_occurrence = (
                        pd.Series(
                            X_input.columns
                        )
                        .groupby(
                            pd.Series(
                                X_input.columns
                            )
                        )
                        .cumcount()
                    )

                    expected_occurrence = (
                        pd.Series(
                            expected_features
                        )
                        .groupby(
                            pd.Series(
                                expected_features
                            )
                        )
                        .cumcount()
                    )

                    # --------------------------------------------------
                    # Create unique feature keys
                    # --------------------------------------------------

                    csv_keys = list(
                        zip(
                            X_input.columns,
                            csv_occurrence
                        )
                    )

                    expected_keys = list(
                        zip(
                            expected_features,
                            expected_occurrence
                        )
                    )

                    # --------------------------------------------------
                    # Check missing features
                    # --------------------------------------------------

                    missing_keys = [
                        key
                        for key in expected_keys
                        if key not in csv_keys
                    ]

                    # --------------------------------------------------
                    # Check extra features
                    # --------------------------------------------------

                    extra_keys = [
                        key
                        for key in csv_keys
                        if key not in expected_keys
                    ]

                    if missing_keys:

                        st.error(
                            "Missing features detected."
                        )

                        st.write(
                            missing_keys
                        )

                        st.stop()

                    if extra_keys:

                        st.error(
                            "Unexpected features detected."
                        )

                        st.write(
                            extra_keys
                        )

                        st.stop()

                    # --------------------------------------------------
                    # Reorder using column positions
                    # --------------------------------------------------

                    column_positions = [
                        csv_keys.index(key)
                        for key in expected_keys
                    ]

                    X_input = X_input.iloc[
                        :,
                        column_positions
                    ]

                else:

                    # --------------------------------------------------
                    # Fallback if scaler has no feature names
                    # --------------------------------------------------

                    if X_input.shape[1] != (
                        scaler.n_features_in_
                    ):

                        st.error(
                            f"Scaler expects "
                            f"{scaler.n_features_in_} "
                            f"features, but received "
                            f"{X_input.shape[1]}."
                        )

                        st.stop()

                # --------------------------------------------------
                # Convert to NumPy
                #
                # This avoids sklearn feature-name checking
                # after we have already manually aligned the
                # duplicate columns.
                # --------------------------------------------------

                X_array = X_input.to_numpy()

                # --------------------------------------------------
                # Final safety check
                # --------------------------------------------------

                if X_array.shape[1] != 561:

                    st.error(
                        f"Final feature count is "
                        f"{X_array.shape[1]}, but the model "
                        f"requires 561."
                    )

                    st.stop()

                # --------------------------------------------------
                # Scale
                # --------------------------------------------------

                X_scaled = scaler.transform(
                    X_array
                )

                # --------------------------------------------------
                # Selected model
                # --------------------------------------------------

                model = models[
                    selected_model
                ]

                # --------------------------------------------------
                # Predictions
                # --------------------------------------------------

                predictions = model.predict(
                    X_scaled
                )

                # --------------------------------------------------
                # Convert predictions to names
                # --------------------------------------------------

                prediction_names = [

                    activity_mapping.get(
                        int(pred),
                        str(pred)
                    )

                    for pred in predictions

                ]

            # --------------------------------------------------
            # Prediction completed
            # --------------------------------------------------

            st.success(
                "Prediction completed successfully!"
            )

            # --------------------------------------------------
            # Prediction Results
            # --------------------------------------------------

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

            # --------------------------------------------------
            # Activity Distribution
            # --------------------------------------------------

            st.subheader(
                "Activity Distribution"
            )

            distribution_df = pd.DataFrame({

                "Activity":
                    prediction_counts.index,

                "Count":
                    prediction_counts.values

            })

            st.bar_chart(
                distribution_df.set_index(
                    "Activity"
                )
            )

            # --------------------------------------------------
            # Detailed Predictions
            # --------------------------------------------------

            st.subheader(
                "Detailed Predictions"
            )

            prediction_output = pd.DataFrame({

                "Predicted Activity":
                    prediction_names

            })

            # --------------------------------------------------
            # Compare actual labels if available
            # --------------------------------------------------

            if "Activity" in data.columns:

                actual_names = [

                    activity_mapping.get(
                        int(value),
                        str(value)
                    )

                    for value in data[
                        "Activity"
                    ]

                ]

                prediction_output.insert(
                    0,
                    "Actual Activity",
                    actual_names
                )

                prediction_output[
                    "Correct"
                ] = (

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

            # --------------------------------------------------
            # Model Performance
            # --------------------------------------------------

            st.markdown("---")

            st.subheader(
                "📊 Model Performance"
            )

            results_file = "model_results.csv"

            if os.path.exists(
                results_file
            ):

                results = pd.read_csv(
                    results_file
                )

                model_result = results[
                    results["Model"]
                    ==
                    selected_model
                ]

                if not model_result.empty:

                    row = model_result.iloc[0]

                    col1, col2, col3 = (
                        st.columns(3)
                    )

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

                    col4, col5, col6 = (
                        st.columns(3)
                    )

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

            # --------------------------------------------------
            # Download predictions
            # --------------------------------------------------

            prediction_csv = (
                prediction_output.to_csv(
                    index=False
                )
            )

            st.download_button(
                label="⬇️ Download Predictions",
                data=prediction_csv,
                file_name=(
                    "activity_predictions.csv"
                ),
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            f"An error occurred while processing "
            f"the file: {e}"
        )

else:

    st.info(
        "Please upload test_data.csv to begin "
        "classification."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Human Activity Recognition | "
    "Machine Learning Classification Project"
)
