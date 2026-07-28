# ==========================================================
# Loan Approval Prediction Dashboard
# visualization.py (Part A)
# ==========================================================

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# Target Distribution Chart
# ==========================================================

def target_distribution_chart(df):

    counts = df["Loan_Status"].value_counts()

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(
        counts.index.astype(str),
        counts.values
    )

    ax.set_title("Loan Status Distribution")

    ax.set_xlabel("Loan Status")

    ax.set_ylabel("Count")

    st.pyplot(fig)


# ==========================================================
# Missing Values Chart
# ==========================================================

def missing_values_chart(df):

    missing = df.isnull().sum()

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        missing.index,
        missing.values
    )

    ax.set_title("Missing Values")

    ax.set_ylabel("Count")

    plt.xticks(rotation=45)

    st.pyplot(fig)


# ==========================================================
# Correlation Heatmap
# ==========================================================

def correlation_heatmap(df):

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8,6))

    image = ax.imshow(
        corr,
        aspect="auto"
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(
        corr.columns,
        rotation=90
    )

    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(
        corr.columns
    )

    plt.colorbar(image)

    st.pyplot(fig)

# ==========================================================
# Accuracy Comparison Chart
# ==========================================================

def accuracy_chart(results_df):

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    ax.set_title("Accuracy Comparison")
    ax.set_ylabel("Accuracy")

    plt.xticks(rotation=20)

    st.pyplot(fig)


# ==========================================================
# Precision Comparison Chart
# ==========================================================

def precision_chart(results_df):

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        results_df["Model"],
        results_df["Precision"]
    )

    ax.set_title("Precision Comparison")
    ax.set_ylabel("Precision")

    plt.xticks(rotation=20)

    st.pyplot(fig)


# ==========================================================
# Recall Comparison Chart
# ==========================================================

def recall_chart(results_df):

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        results_df["Model"],
        results_df["Recall"]
    )

    ax.set_title("Recall Comparison")
    ax.set_ylabel("Recall")

    plt.xticks(rotation=20)

    st.pyplot(fig)


# ==========================================================
# F1 Score Comparison Chart
# ==========================================================

def f1_chart(results_df):

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        results_df["Model"],
        results_df["F1 Score"]
    )

    ax.set_title("F1 Score Comparison")
    ax.set_ylabel("F1 Score")

    plt.xticks(rotation=20)

    st.pyplot(fig)


# ==========================================================
# Training Time Comparison Chart
# ==========================================================

def training_time_chart(results_df):

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        results_df["Model"],
        results_df["Training Time"]
    )

    ax.set_title("Training Time Comparison")
    ax.set_ylabel("Seconds")

    plt.xticks(rotation=20)

    st.pyplot(fig)


# ==========================================================
# Model Comparison Chart
# ==========================================================

def model_comparison_chart(results_df):

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        results_df["Model"],
        results_df["Accuracy"],
        marker="o",
        label="Accuracy"
    )

    ax.plot(
        results_df["Model"],
        results_df["Precision"],
        marker="o",
        label="Precision"
    )

    ax.plot(
        results_df["Model"],
        results_df["Recall"],
        marker="o",
        label="Recall"
    )

    ax.plot(
        results_df["Model"],
        results_df["F1 Score"],
        marker="o",
        label="F1 Score"
    )

    ax.set_title("Model Performance Comparison")

    ax.legend()

    plt.xticks(rotation=20)

    st.pyplot(fig)


# ==========================================================
# Confusion Matrix Chart
# ==========================================================

def confusion_matrix_chart(cm):

    fig, ax = plt.subplots(figsize=(5,5))

    image = ax.imshow(cm)

    plt.colorbar(image)

    ax.set_title("Confusion Matrix")

    ax.set_xlabel("Predicted")

    ax.set_ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(
                j,
                i,
                cm[i][j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)