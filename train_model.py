import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def map_dependency_levels(target_series):
    dependency_label_mapping = {
        1: "Low",
        2: "Low",
        3: "Low",
        4: "Medium",
        5: "Medium",
        6: "Medium",
        7: "Medium",
        8: "High",
        9: "High",
        10: "High",
    }
    mapped_target = target_series.map(dependency_label_mapping)
    if mapped_target.isna().any():
        invalid_values = sorted(target_series[mapped_target.isna()].unique().tolist())
        raise ValueError(
            "Perceived_AI_Dependency contains unmapped values: "
            f"{invalid_values}. Expected values are integers 1 through 10."
        )
    return mapped_target


def select_model_features(data_df):
    model_features = [
        "Year_of_Study",
        "Weekly_GenAI_Hours",
        "Traditional_Study_Hours",
        "Anxiety_Level_During_Exams",
        "Skill_Retention_Score",
        "Institutional_Policy",
    ]
    return data_df[model_features]


def train_and_evaluate(data_path="preprocessed_data.csv"):
    # Load the cleaned dataset produced by features.py
    data_df = pd.read_csv(data_path)

    # Choose the column you want to predict as the target
    target_column = "Perceived_AI_Dependency"  # classification target for logistic regression

    # Keep only the strongest predictors from the first model run
    feature_df = select_model_features(data_df)

    target_series = map_dependency_levels(data_df[target_column])

    # Split before fitting encoders/scalers
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        target_series,
        test_size=0.2,
        random_state=42,
        stratify=target_series,
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Apply log transform on the train data and the same transform on test
    X_train["Weekly_GenAI_Hours"] = np.log1p(X_train["Weekly_GenAI_Hours"])
    X_test["Weekly_GenAI_Hours"] = np.log1p(X_test["Weekly_GenAI_Hours"])

    categorical_cols = ["Institutional_Policy"]

    encoder = OneHotEncoder(drop="first", sparse_output=False, dtype=int, handle_unknown="ignore")
    scaler = MinMaxScaler()

    X_train_cat = pd.DataFrame(
        np.asarray(encoder.fit_transform(X_train[categorical_cols])),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_train.index,
    )
    X_test_cat = pd.DataFrame(
        np.asarray(encoder.transform(X_test[categorical_cols])),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_test.index,
    )

    scaled_columns = [
        "Year_of_Study",
        "Weekly_GenAI_Hours",
        "Traditional_Study_Hours",
        "Anxiety_Level_During_Exams",
        "Skill_Retention_Score",
    ]
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[scaled_columns]),
        columns=scaled_columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[scaled_columns]),
        columns=scaled_columns,
        index=X_test.index,
    )

    X_train_processed = pd.concat([X_train_scaled, X_train_cat], axis=1)
    X_test_processed = pd.concat([X_test_scaled, X_test_cat], axis=1)

    # Baseline: majority class classifier
    majority_class = y_train.value_counts().idxmax()
    baseline_accuracy = (y_test == majority_class).mean()
    print(f"Baseline accuracy (majority class '{majority_class}'): {baseline_accuracy:.4f}")

    # Train logistic regression on Perceived_AI_Dependency
    default_model = LogisticRegression(max_iter=1000)
    default_model.fit(X_train_processed, y_train)
    y_pred = default_model.predict(X_test_processed)
    model_accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("\nLogistic regression performance:")
    print("Accuracy:", model_accuracy)
    print(class_report)

    return {
        "baseline_accuracy": baseline_accuracy,
        "accuracy": model_accuracy,
        "classification_report": class_report,
        "train_rows": len(X_train_processed),
        "test_rows": len(X_test_processed),
        "feature_columns": X_train_processed.columns.tolist(),
    }


if __name__ == "__main__":
    train_and_evaluate()