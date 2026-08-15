import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_correlation_with_target(corr_series, output_path="assets/correlation_with_burnout.png"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    corr_series.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Absolute correlation with Burnout_Risk_Level")
    ax.set_title("Numeric feature correlation with Burnout_Risk_Level")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved correlation chart to {output_path}")


def run_eda():
    # Load preprocessed data (run features.py first)
    data_df = pd.read_csv("preprocessed_data.csv")

    # Encode Burnout_Risk_Level as ordinal for correlation analysis
    risk_mapping = {"Low": 1, "Medium": 2, "High": 3}
    data_df["Burnout_Risk_Level_Code"] = data_df["Burnout_Risk_Level"].map(risk_mapping)

    # Correlation: numeric features vs. burnout level
    numeric_cols = data_df.select_dtypes(include=["number"]).columns.tolist()
    if "Burnout_Risk_Level_Code" in numeric_cols:
        target_series = data_df["Burnout_Risk_Level_Code"]
        corr_with_target = (
            data_df[numeric_cols]
            .corrwith(target_series)
            .abs()
            .sort_values(ascending=False)
        )
        top_corr = corr_with_target.drop("Burnout_Risk_Level_Code").head(10)
        print("\nTop numeric features correlated with Burnout_Risk_Level:")
        print(top_corr)
        plot_correlation_with_target(top_corr)
    else:
        print("\nBurnout_Risk_Level_Code not found in numeric columns.")

    # Categorical feature analysis: average burnout risk by category
    categorical_features = [
        "Major_Category",
        "Primary_Use_Case",
        "Prompt_Engineering_Skill",
        "Institutional_Policy",
    ]
    print("\nCategorical features sorted by average burnout risk:")
    for cat in categorical_features:
        if cat in data_df.columns:
            summary = (
                data_df
                .groupby(cat)["Burnout_Risk_Level_Code"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "avg_risk_code", "count": "frequency"})
                .sort_values(by="avg_risk_code", ascending=False)
            )
            print(f"\n{cat}:")
            print(summary.head(5))
        else:
            print(f"\n{cat} not found in dataset")


if __name__ == "__main__":
    run_eda()
