import tempfile
import unittest
from pathlib import Path

import pandas as pd

from features import preprocess
from train_model import map_dependency_levels, train_and_evaluate


class TestFeatureEngineeringPipeline(unittest.TestCase):
    def test_preprocess_outputs_expected_schema_and_mapped_year(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_path = tmp_path / "raw.csv"
            out_path = tmp_path / "clean.csv"

            raw_df = pd.DataFrame(
                {
                    "Student_ID": [1, 2, 2, 3],
                    "Year_of_Study": ["Freshman", "Junior", "Junior", "Senior"],
                    "Perceived_AI_Dependency": [1, 6, 6, 9],
                    "Weekly_GenAI_Hours": [1, 3, 3, 5],
                    "Traditional_Study_Hours": [10, 8, 8, 7],
                    "Anxiety_Level_During_Exams": [4, 5, 5, 7],
                    "Skill_Retention_Score": [8, 7, 7, 6],
                    "Institutional_Policy": [
                        "Allowed_With_Citation",
                        "Strict_Ban",
                        "Strict_Ban",
                        "Actively_Encouraged",
                    ],
                }
            )
            raw_df.to_csv(raw_path, index=False)

            cleaned_df = preprocess(input_path=str(raw_path), output_path=str(out_path))

            self.assertTrue(out_path.exists())
            self.assertEqual(cleaned_df["Year_of_Study"].tolist(), [1, 3, 4])
            self.assertIn("Perceived_AI_Dependency", cleaned_df.columns)

    def test_dependency_mapping_rejects_unexpected_values(self):
        dependency_series = pd.Series([1, 5, 11])
        with self.assertRaises(ValueError):
            map_dependency_levels(dependency_series)

    def test_training_pipeline_fits_and_predicts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "preprocessed_data.csv"

            rows = []
            classes = [2, 6, 9] * 10
            policies = ["Allowed_With_Citation", "Strict_Ban", "Actively_Encouraged"] * 10

            for i, (dep, policy) in enumerate(zip(classes, policies), start=1):
                rows.append(
                    {
                        "Student_ID": i,
                        "Year_of_Study": (i % 5) + 1,
                        "Weekly_GenAI_Hours": float((i % 12) + 1),
                        "Traditional_Study_Hours": float(5 + (i % 10)),
                        "Anxiety_Level_During_Exams": float((i % 10) + 1),
                        "Skill_Retention_Score": float(10 - (i % 10)),
                        "Institutional_Policy": policy,
                        "Perceived_AI_Dependency": dep,
                    }
                )

            pd.DataFrame(rows).to_csv(data_path, index=False)

            metrics = train_and_evaluate(data_path=str(data_path))

            self.assertGreaterEqual(metrics["accuracy"], 0.0)
            self.assertLessEqual(metrics["accuracy"], 1.0)
            self.assertGreater(metrics["train_rows"], 0)
            self.assertGreater(metrics["test_rows"], 0)
            self.assertIn("Institutional_Policy_Strict_Ban", metrics["feature_columns"])


if __name__ == "__main__":
    unittest.main()
