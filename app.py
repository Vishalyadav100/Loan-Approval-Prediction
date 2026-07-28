# ==========================================================
# Loan Approval Prediction Dashboard
# app.py (Part A)
# ==========================================================

# -----------------------------
# Import Libraries
# -----------------------------
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Import Custom Modules
# -----------------------------
from preprocessing import (
    preprocess_pipeline,
    dataset_summary,
    missing_values,
    encode_data,
    drop_columns,
    fill_missing_values
)

from models import (
    get_models,
    train_models,
    result_dataframe,
    best_model,
    save_best_model,
    get_confusion_matrix
)

from prediction import (
    customer_input,
    encode_prediction,
    predict_with_error_handling
)

from visualization import (
    accuracy_chart,
    precision_chart,
    recall_chart,
    f1_chart,
    training_time_chart,
    target_distribution_chart,
    missing_values_chart,
    correlation_heatmap,
    model_comparison_chart,
    confusion_matrix_chart
)

from download import (
    download_results,
    download_model,
    download_processed_dataset,
    download_project_info
)

from utils import (
    footer,
    about_project,
    project_information
)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Loan Approval Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Custom CSS
# ==========================================================

st.markdown("""
<style>

/* Main Container */

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:2rem;
    padding-right:2rem;
}

/* Metric Cards */

div[data-testid="metric-container"]{
    background:#ffffff;
    border-radius:15px;
    padding:18px;
    border:1px solid #d9d9d9;
    box-shadow:0 3px 10px rgba(0,0,0,.08);
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background:#eef5ff;
}

/* Buttons */

.stButton>button{

    width:100%;
    border-radius:10px;
    height:3em;
    font-size:16px;
}

/* Download Button */

.stDownloadButton>button{

    width:100%;
    border-radius:10px;
    height:3em;
}

/* Dataframe */

[data-testid="stDataFrame"]{

    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# Load Dataset
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/loan_approval_dataset.csv"
    )

    return df


df = load_data()

# ==========================================================
# Session State
# ==========================================================

if "results_df" not in st.session_state:
    st.session_state["results_df"] = None

if "best_model" not in st.session_state:
    st.session_state["best_model"] = None

if "trained_models" not in st.session_state:
    st.session_state["trained_models"] = None

# ==========================================================
# Dashboard Title
# ==========================================================

st.title("🏦 Loan Approval Prediction Dashboard")

st.caption(
    "Predict Loan Approval using Machine Learning"
)

st.divider()

# ==========================================================
# Dashboard Summary
# ==========================================================

summary = dataset_summary(df)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Dataset Rows",
        summary["rows"]
    )

with col2:

    st.metric(
        "Features",
        summary["features"]
    )

with col3:

    st.metric(
        "ML Models",
        "4"
    )

with col4:

    if st.session_state["results_df"] is None:

        st.metric(
            "Best Accuracy",
            "--"
        )

    else:

        acc = st.session_state["results_df"]["Accuracy"].max()

        st.metric(
            "Best Accuracy",
            f"{acc:.4f}"
        )

st.divider()

# ==========================================================
# Sidebar Navigation
# ==========================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/bank-building.png",
    width=80
)

st.sidebar.title("Navigation")

menu = st.sidebar.radio(

    "Select Module",

    [

        "🏠 Home",

        "📂 Dataset Analysis",

        "🧹 Data Preprocessing",

        "🤖 Model Training",

        "📊 Model Comparison",

        "🔮 Loan Prediction",

        "📈 Visualization",

        "⬇ Download",

        "ℹ About"

    ]

)

st.sidebar.divider()

st.sidebar.success(
    "Loan Approval Prediction Dashboard"
)

st.sidebar.caption(
    "Developed using Streamlit + Scikit-Learn"
)

# ==========================================================
# Remaining Sections
# (Part B starts from here)
# ==========================================================


# ==========================================================
# HOME PAGE
# ==========================================================

if menu == "🏠 Home":

    st.header("🏠 Home")

    st.write(
        """
Welcome to the **Loan Approval Prediction Dashboard**.

This application allows you to:

✅ Analyze the Dataset

✅ Perform Data Preprocessing

✅ Train Multiple Machine Learning Models

✅ Compare Model Performance

✅ Predict Loan Approval

✅ Download the Best Model
"""
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Dataset Rows",
            df.shape[0]
        )

        st.metric(
            "Dataset Columns",
            df.shape[1]
        )

    with col2:

        st.metric(
            "Target Column",
            "Loan_Status"
        )

        st.metric(
            "Algorithms",
            "4"
        )

    st.success(
        "Select a module from the left sidebar to begin."
    )

# ==========================================================
# DATASET ANALYSIS
# ==========================================================

elif menu == "📂 Dataset Analysis":

    st.header("📂 Dataset Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    st.divider()

    st.subheader("Missing Values")

    missing_df = missing_values(df)

    st.dataframe(missing_df)

    missing_values_chart(df)

    st.divider()

    st.subheader("Loan Status Distribution")

    target_distribution_chart(df)

    st.divider()

    st.subheader("Correlation Heatmap")

    temp_df = encode_data(
        drop_columns(
            fill_missing_values(
                df.copy()
            )
        )
    )

    correlation_heatmap(temp_df)

# ==========================================================
# DATA PREPROCESSING
# ==========================================================

elif menu == "🧹 Data Preprocessing":

    st.header("🧹 Data Preprocessing")

    if "Loan_Status" not in df.columns:

        st.error(
            "Target Column not found."
        )

        st.stop()

    target = "Loan_Status"

    st.success(
        f"Target Column : {target}"
    )

    test_size = st.slider(

        "Test Size",

        0.10,

        0.50,

        0.20

    )

    random_state = st.number_input(

        "Random State",

        value=42

    )

    with st.spinner(

        "Running Preprocessing..."

    ):

        (

            processed_df,

            X_train,

            X_test,

            y_train,

            y_test,

            feature_names

        ) = preprocess_pipeline(

            df,

            target,

            test_size,

            random_state

        )

    st.success(
        "Preprocessing Completed Successfully."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Training Samples",

            len(X_train)

        )

    with col2:

        st.metric(

            "Testing Samples",

            len(X_test)

        )

    with col3:

        st.metric(

            "Features",

            len(feature_names)

        )

    st.divider()

    st.subheader(
        "Processed Dataset"
    )

    st.dataframe(
        processed_df.head()
    )

    st.session_state["processed_df"] = processed_df

    st.session_state["X_train"] = X_train

    st.session_state["X_test"] = X_test

    st.session_state["y_train"] = y_train

    st.session_state["y_test"] = y_test

    st.session_state["feature_names"] = feature_names


# ==========================================================
# MODEL TRAINING
# ==========================================================

elif menu == "🤖 Model Training":

    st.header("🤖 Model Training")

    if st.button("🚀 Train All Models", use_container_width=True):

        with st.spinner("Training Models..."):

            if "X_train" not in st.session_state:

                st.warning(
                    "Please run Data Preprocessing first."
                )

                st.stop()

            X_train = st.session_state["X_train"]
            X_test = st.session_state["X_test"]
            y_train = st.session_state["y_train"]
            y_test = st.session_state["y_test"]

            models = get_models()

            results, trained_models = train_models(

                models,

                X_train,

                X_test,

                y_train,

                y_test

            )

            results_df = result_dataframe(results)

            st.session_state["results_df"] = results_df

            st.session_state["trained_models"] = trained_models

            best_model_name = best_model(results_df)

            st.session_state["best_model"] = best_model_name

            save_best_model(

                trained_models,

                best_model_name

            )

            st.success("✅ All Models Trained Successfully")

            st.subheader("Training Results")

            st.dataframe(results_df)

            st.success(

                f"🏆 Best Model : {best_model_name}"

            )

    else:

        st.info(

            "Click 'Train All Models' to begin training."

        )


# ==========================================================
# MODEL COMPARISON
# ==========================================================

elif menu == "📊 Model Comparison":

    st.header("📊 Model Comparison")

    if st.session_state["results_df"] is None:

        st.warning("Please train the models first.")

    else:

        results_df = st.session_state["results_df"]

        st.subheader("Model Performance")

        st.dataframe(results_df)

        st.divider()

        best = results_df.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Best Accuracy",
                round(best["Accuracy"],4)
            )

        with col2:
            st.metric(
                "Best Precision",
                round(best["Precision"],4)
            )

        with col3:
            st.metric(
                "Best Recall",
                round(best["Recall"],4)
            )

        with col4:
            st.metric(
                "Best F1 Score",
                round(best["F1 Score"],4)
            )

        st.divider()

        st.subheader("Accuracy Comparison")
        accuracy_chart(results_df)

        st.subheader("Precision Comparison")
        precision_chart(results_df)

        st.subheader("Recall Comparison")
        recall_chart(results_df)

        st.subheader("F1 Score Comparison")
        f1_chart(results_df)

        st.subheader("Training Time Comparison")
        training_time_chart(results_df)

        st.subheader("Overall Model Comparison")
        model_comparison_chart(results_df)

        st.divider()

        st.subheader("Best Model")

        st.success(
            f"""
🏆 {best['Model']}

Accuracy : {best['Accuracy']:.4f}
"""
        )

        trained_models = st.session_state["trained_models"]

        cm = get_confusion_matrix(

            trained_models,

            best["Model"]

        )

        st.subheader("Confusion Matrix")

        confusion_matrix_chart(cm)

        st.divider()

        download_results(results_df)

# ==========================================================
# VISUALIZATION
# ==========================================================

elif menu == "📈 Visualization":

    st.header("📈 Visualization")

    if st.session_state["results_df"] is None:

        st.warning(
            "Please train the models first."
        )

    else:

        results_df = st.session_state["results_df"]

        st.subheader("Accuracy")
        accuracy_chart(results_df)

        st.subheader("Precision")
        precision_chart(results_df)

        st.subheader("Recall")
        recall_chart(results_df)

        st.subheader("F1 Score")
        f1_chart(results_df)

        st.subheader("Training Time")
        training_time_chart(results_df)

        st.subheader("Overall Comparison")
        model_comparison_chart(results_df)



# ==========================================================
# LOAN PREDICTION
# ==========================================================

elif menu == "🔮 Loan Prediction":

    st.header("🔮 Loan Approval Prediction")

    input_df = customer_input()

    st.subheader("Applicant Details")

    st.dataframe(input_df)

    encoded_df = encode_prediction(input_df)

    if st.button(
        "Predict Loan Approval",
        use_container_width=True
    ):

        predict_with_error_handling(encoded_df)

# ==========================================================
# DOWNLOAD CENTER
# ==========================================================

elif menu == "⬇ Download":

    st.header("⬇ Download Center")

    if st.session_state["results_df"] is None:

        st.warning(
            "Please train the models first."
        )

    else:

        results_df = st.session_state["results_df"]

        processed_df = st.session_state["processed_df"]

        st.subheader("Download Files")

        download_results(results_df)

        download_processed_dataset(
            processed_df
        )

        download_model()

        download_project_info()

# ==========================================================
# ABOUT
# ==========================================================

elif menu == "ℹ About":

    about_project()

    project_information()

# ==========================================================
# FOOTER
# ==========================================================

footer()