# ==========================================================
# Loan Approval Prediction Dashboard
# models.py (Part A)
# ==========================================================

import time
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    roc_auc_score

)


# ==========================================================
# Available Models
# ==========================================================

def get_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(

            n_estimators=100,

            random_state=42

        ),

        "XGBoost": XGBClassifier(

            random_state=42,

            eval_metric="logloss"

        )

    }

    return models


# ==========================================================
# Train All Models
# ==========================================================

def train_models(

    models,

    X_train,

    X_test,

    y_train,

    y_test

):

    results = []

    trained_models = {}

    for name, model in models.items():

        start = time.time()

        model.fit(

            X_train,

            y_train

        )

        end = time.time()

        prediction = model.predict(

            X_test

        )

        training_time = round(

            end-start,

            4

        )

        accuracy = accuracy_score(

            y_test,

            prediction

        )

        precision = precision_score(

            y_test,

            prediction

        )

        recall = recall_score(

            y_test,

            prediction

        )

        f1 = f1_score(

            y_test,

            prediction

        )

        cm = confusion_matrix(

            y_test,

            prediction

        )

        trained_models[name] = {

            "model": model,

            "confusion_matrix": cm

        }

        results.append({

            "Model": name,

            "Accuracy": round(

                accuracy,

                4

            ),

            "Precision": round(

                precision,

                4

            ),

            "Recall": round(

                recall,

                4

            ),

            "F1 Score": round(

                f1,

                4

            ),

            "Training Time": training_time,

            "AUC": round(calculate_auc(model,X_test,y_test),4) if hasattr(model,"predict_proba") else None})

    return results, trained_models


# ==========================================================
# Results DataFrame
# ==========================================================

def result_dataframe(results):

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(

        by="Accuracy",

        ascending=False

    ).reset_index(drop=True)

    return results_df


# ==========================================================
# Best Model Name
# ==========================================================

def best_model(results_df):

    return results_df.iloc[0]["Model"]


# ==========================================================
# Best Model Object
# ==========================================================

def best_model_object(

    trained_models,

    best_model_name

):

    return trained_models[best_model_name]["model"]


# ==========================================================
# Confusion Matrix
# ==========================================================

def get_confusion_matrix(

    trained_models,

    best_model_name

):

    return trained_models[best_model_name][

        "confusion_matrix"

    ]


# ==========================================================
# ROC-AUC Score
# ==========================================================

def calculate_auc(

    model,

    X_test,

    y_test

):

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(

            X_test

        )[:,1]

        auc = roc_auc_score(

            y_test,

            probability

        )

        return round(

            auc,

            4

        )

    return None


# ==========================================================
# Model Ranking
# ==========================================================

def model_ranking(results_df):

    ranking = results_df.copy()

    ranking.insert(

        0,

        "Rank",

        range(

            1,

            len(ranking)+1

        )

    )

    return ranking


# ==========================================================
# Save Best Model
# ==========================================================

def save_best_model(

    trained_models,

    best_model_name,

    path="saved_models/Best_Model.pkl"

):

    import os
    import joblib

    os.makedirs(

        "saved_models",

        exist_ok=True

    )

    joblib.dump(

        trained_models[best_model_name]["model"],

        path

    )