import pandas as pd


def map_year_of_study(year_series):
    year_mapping = {
        "Freshman": 1,
        "Sophomore": 2,
        "Junior": 3,
        "Senior": 4,
        "Graduate": 5,
    }
    return year_series.map(year_mapping)


def preprocess(input_path="ai_student_impact_dataset.csv", output_path="preprocessed_data.csv"):
    # Load raw dataset
    data_df = pd.read_csv(input_path)

    # Remove any duplicate or null values
    data_df = data_df.drop_duplicates()
    data_df = data_df.dropna()

    # Map Year_of_Study to ordinal values
    data_df["Year_of_Study"] = map_year_of_study(data_df["Year_of_Study"])

    # Save the cleaned dataset for later use
    data_df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path}")
    return data_df


if __name__ == "__main__":
    preprocess()
