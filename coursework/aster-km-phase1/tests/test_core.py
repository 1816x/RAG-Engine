import unittest

from km_proto.core import CATEGORIES, classify, evaluate, load_documents, search


class PrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_documents()

    def test_corpus_has_balanced_categories_and_unique_ids(self):
        self.assertEqual(len(self.docs), 32)
        self.assertEqual(len({d.id for d in self.docs}), 32)
        for category in CATEGORIES:
            self.assertEqual(sum(d.category == category for d in self.docs), 8)

    def test_holdout_evaluation_has_no_training_leak(self):
        result = evaluate(self.docs)
        self.assertEqual((result["train_count"], result["test_count"]), (24, 8))
        self.assertEqual(len(result["confusion_matrix"]), 4)
        self.assertGreaterEqual(result["accuracy"], 0.75)

    def test_search_ranks_relevant_test_procedure(self):
        hits = search("steps to reproduce session timeout test expected result", self.docs, 3)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0].category, "Test Procedure")
        self.assertGreater(hits[0][1], 0)

    def test_new_document_classification(self):
        category, score = classify(
            "The incident reported by support affected the display; logs show an error and priority is high.",
            self.docs,
        )
        self.assertEqual(category, "Incident Report")
        self.assertGreater(score, 0)

    def test_empty_search_rejected(self):
        with self.assertRaises(ValueError):
            search(" ", self.docs)


if __name__ == "__main__":
    unittest.main()

