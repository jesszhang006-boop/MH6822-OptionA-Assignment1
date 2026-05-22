import subprocess
import unittest
from pathlib import Path

import numpy as np

import build_assignment as ba


class Task3GenerationTests(unittest.TestCase):
    def test_application_scores_use_trained_logistic_predictions(self):
        df = ba.generate_data(n=600)
        model, _metrics = ba.train_model(df)

        self.assertTrue(
            hasattr(ba, "score_applications"),
            "generation pipeline should expose score_applications(model, df)",
        )
        scored = ba.score_applications(model, df)
        expected = np.round(model.predict_proba(scored[ba.MODEL_FEATURES])[:, 1], 4)

        np.testing.assert_allclose(scored["model_score"].to_numpy(), expected, atol=0.0001)
        self.assertTrue(((scored["approved"] == 1) == (scored["model_score"] < ba.APPROVAL_THRESHOLD)).all())
        self.assertTrue(
            (
                (scored["manual_review"] == 1)
                == (
                    (scored["model_score"] >= ba.MANUAL_REVIEW_LOWER)
                    & (scored["model_score"] < ba.MANUAL_REVIEW_UPPER)
                )
            ).all()
        )

    def test_eu_threshold_is_marked_as_internal_governance_not_law(self):
        rules = ba.jurisdiction_rules()
        self.assertIn("threshold_basis", rules.columns)
        eu_basis = rules.loc[rules["jurisdiction"].eq("EU"), "threshold_basis"].iloc[0]

        self.assertIn("not a statutory threshold", eu_basis.lower())
        self.assertIn("internal governance", eu_basis.lower())

    def test_eu_obligation_matrix_and_data_dictionary_are_defined(self):
        self.assertTrue(hasattr(ba, "eu_ai_act_obligations"))
        self.assertTrue(hasattr(ba, "data_dictionary"))

        obligations = ba.eu_ai_act_obligations()
        dictionary = ba.data_dictionary()

        self.assertEqual(
            [
                "obligation_id",
                "jurisdiction",
                "source",
                "article",
                "control",
                "evidence_output",
                "owner",
            ],
            list(obligations.columns),
        )
        self.assertTrue({"EU-AIA-10-01", "EU-AIA-14-01", "EU-AIA-17-01"}.issubset(set(obligations["obligation_id"])))
        self.assertIn("used_in_model", dictionary.columns)
        self.assertIn("protected_or_proxy", dictionary.columns)
        self.assertIn("production_credit_model_note", dictionary.columns)
        self.assertFalse(dictionary.loc[dictionary["field_name"].eq("protected_group"), "used_in_model"].iloc[0])

    def test_main_deck_is_compressed_to_10_to_12_pages(self):
        deck_path = ba.ROOT / "Task3_Tool_Design_Deck.pdf"
        output = subprocess.check_output(["pdfinfo", str(deck_path)], text=True)
        pages_line = next(line for line in output.splitlines() if line.startswith("Pages:"))
        pages = int(pages_line.split(":", 1)[1].strip())

        self.assertGreaterEqual(pages, 10)
        self.assertLessEqual(pages, 12)

    def test_zip_contains_executed_notebook_and_new_governance_csvs(self):
        self.assertTrue(hasattr(ba, "EXECUTED_NOTEBOOK_NAME"))
        expected_files = {
            ba.EXECUTED_NOTEBOOK_NAME,
            "data/Task3_eu_ai_act_obligations.csv",
            "data/Task3_data_dictionary.csv",
        }
        import zipfile

        with zipfile.ZipFile(ba.ROOT / ba.ZIP_NAME) as archive:
            names = set(archive.namelist())

        self.assertTrue(expected_files.issubset(names))


if __name__ == "__main__":
    unittest.main()
