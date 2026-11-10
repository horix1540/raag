"""Unit tests for Gate B v2's input construction (model-free).

Gate B v2 changes only what NLI is shown - the hypothesis and the premise -
so those two pure functions are where a silent mistake would change every
score. v1 behaviour ("claim" + "passage") must stay byte-for-byte identical,
because the saved Milestone 2-4 scores are reproduced from it.
"""

from __future__ import annotations

from src.scorers import build_hypothesis, premise_windows

Q = "Where did many Acadians move to?"


class TestBuildHypothesis:
    def test_v1_claim_mode_is_untouched(self):
        assert build_hypothesis("New Orleans [1]", Q, "claim") == "New Orleans [1]"

    def test_no_question_falls_back_to_claim(self):
        assert build_hypothesis("New Orleans [1]", None, "qa") == "New Orleans [1]"

    def test_qa_prefixes_question_and_drops_citation(self):
        assert build_hypothesis("New Orleans [1]", Q, "qa") == f"{Q} New Orleans"

    def test_qa_short_leaves_full_sentences_alone(self):
        long = "Many Acadians moved to New Orleans after being expelled by the British [2]."
        assert build_hypothesis(long, Q, "qa_short") == (
            "Many Acadians moved to New Orleans after being expelled by the British ."
        )
        assert build_hypothesis("New Orleans [1]", Q, "qa_short") == f"{Q} New Orleans"


class TestPremiseWindows:
    def test_short_passage_kept_whole(self):
        assert premise_windows(["One. Two."]) == [("One. Two.", 0)]

    def test_windows_overlap_with_stride_one_and_keep_origin(self):
        ctx = "A one. B two. C three. D four."
        wins = premise_windows(["x.", ctx], size=3)
        assert wins[0] == ("x.", 0)
        assert [w for w, _ in wins[1:]] == ["A one. B two. C three.", "B two. C three. D four."]
        assert all(o == 1 for _, o in wins[1:])
