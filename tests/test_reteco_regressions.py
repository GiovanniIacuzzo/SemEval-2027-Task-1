"""Regression checks for prompt budgeting, conversation splits and TREC rankings."""
import tempfile
import unittest
from pathlib import Path

from subtrack_2a.utils.utils import compute_official_ndcg, ranking_scores, validate_trec_file, write_trec_run
from subtrack_2b.dataset.dataset import (
    GroundedGenerationDataset, GroundedTurnSample, resolve_gold_passages, split_conversations_train_val,
)
from subtrack_2b.generate_submission import validate_predictions


class WordTokenizer:
    chat_template = None
    eos_token_id = 1

    def __init__(self):
        self.ids = {}

    def encode(self, text, **kwargs):
        return [self.ids.setdefault(word, len(self.ids) + 2) for word in text.split()]

    def decode(self, ids, **kwargs):
        words = {token: word for word, token in self.ids.items()}
        return " ".join(words[token] for token in ids)


def sample(domain="drones", conv="ex", turn=1, gold=None):
    return GroundedTurnSample(
        domain=domain, conversation_id=conv, turn_id=turn, topic_id=f"{domain}_{conv}_turn_{turn}",
        query="What is the tailfact?", conversation_history="No previous conversation.",
        gold_doc_ids=["gold"], gold_passages=gold or ["The tailfact is here."], answer="The final answer.",
    )


class GenerationTests(unittest.TestCase):
    def test_adaptive_budget_keeps_fact_at_end(self):
        case = sample(gold=[" ".join(["filler"] * 120 + ["tailfact"])])
        prefix = GroundedGenerationDataset([case], WordTokenizer(), max_input_tokens=230,
                                           is_training=False, evidence_strategy="prefix")
        adaptive = GroundedGenerationDataset([case], WordTokenizer(), max_input_tokens=230,
                                             is_training=False, evidence_strategy="adaptive")
        self.assertTrue(prefix[0]["prompt_diagnostics"]["evidence_truncated"])
        self.assertIn("tailfact", adaptive[0]["prompt_diagnostics"]["evidence_shown"][0])
        self.assertFalse(adaptive[0]["prompt_diagnostics"]["evidence_truncated"])
        self.assertLessEqual(len(adaptive[0]["prompt_ids"]), 230)

    def test_refinement_respects_the_same_budget(self):
        case = sample(gold=[" ".join(["filler"] * 350) + ". The tailfact matters."])
        dataset = GroundedGenerationDataset(
            [case], WordTokenizer(), max_input_tokens=180, is_training=False, evidence_strategy="sentences"
        )
        _, ids, diagnostics = dataset.prepare_prompt(case, draft_answer="A draft.")
        self.assertLessEqual(len(ids), 180)
        self.assertIn("The tailfact matters.", diagnostics["evidence_shown"][0])

    def test_split_by_domain_and_whole_conversation(self):
        cases = [sample(domain, str(conv), turn) for domain in ("drones", "robotics")
                 for conv in range(4) for turn in (1, 2)]
        train, val = split_conversations_train_val(cases)
        self.assertEqual({row.domain for row in val}, {"drones", "robotics"})
        train_keys = {(row.domain, row.conversation_id) for row in train}
        val_keys = {(row.domain, row.conversation_id) for row in val}
        self.assertFalse(train_keys & val_keys)
        self.assertEqual(len(train) + len(val), len(cases))

    def test_manifest_missing_gold_is_distinguished_from_broken_lookup(self):
        diagnostics = {}
        ids = {"gold_doc_ids": ["found", "documented", "unexpected"]}
        passages = resolve_gold_passages(ids, {"found": "Passage"}, diagnostics, frozenset({"documented"}))
        self.assertEqual(passages, ["Passage"])
        self.assertEqual(diagnostics["documented_missing_gold_ids_count"], 1)
        self.assertEqual(diagnostics["missing_doc_ids_set"], {"unexpected"})

    def test_reject_missing_turn_or_empty_answer(self):
        records = [{"turn_id": "a", "answer": "Answer"}]
        validate_predictions(records, {"a"})
        with self.assertRaises(ValueError):
            validate_predictions(records, {"a", "b"})
        with self.assertRaises(ValueError):
            validate_predictions([{"turn_id": "a", "answer": " "}], {"a"})


class RetrievalTests(unittest.TestCase):
    def test_writer_and_scorer_use_the_same_order(self):
        run = {"a": [("gold", 0.2), ("gold", 0.2), ("other", 0.9)]}
        qrels = {"a": {"gold": 1}, "b": {"missing": 1}}
        run_scores = {topic: ranking_scores(ranking) for topic, ranking in run.items()}
        self.assertEqual(compute_official_ndcg(qrels, run_scores)["ndcg_cut_10"], 0.5)
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "submission.trec"
            write_trec_run(run, file, max_k=2)
            self.assertEqual([line.split()[2] for line in file.read_text().splitlines()], ["gold", "other"])
            self.assertTrue(validate_trec_file(file)[0])


if __name__ == "__main__":
    unittest.main()
