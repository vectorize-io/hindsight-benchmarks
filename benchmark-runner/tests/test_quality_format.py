from hindsight_benchmark.quality import (
    _format_context,
    _format_grading_material,
    count_tokens,
)


class FakeFact:
    def __init__(self, text, context=None, occurred_start=None, occurred_end=None, mentioned_at=None):
        self.text = text
        self.context = context
        self.occurred_start = occurred_start
        self.occurred_end = occurred_end
        self.mentioned_at = mentioned_at


class FakeRecall:
    def __init__(self, results):
        self.results = results


def test_format_context_contains_facts_and_no_source_block():
    recall = FakeRecall(
        results=[FakeFact("Melanie ran a race", context="chat session 2", mentioned_at="2023-05-20")],
    )
    out = _format_context(recall)
    assert "Fact 1: Melanie ran a race" in out
    assert "Context: chat session 2" in out
    assert "Mentioned: 2023-05-20" in out
    # The whole point of facts-only: raw chunks never reach the generator.
    assert "Source" not in out


def test_format_context_occurred_range():
    recall = FakeRecall(results=[FakeFact("Trip", occurred_start="2024-01-01", occurred_end="2024-01-05")])
    out = _format_context(recall)
    assert "Occurred: 2024-01-01 to 2024-01-05" in out


def test_format_context_empty():
    assert _format_context(FakeRecall(results=[])) == "No memories found."


def test_grading_material_answer_only():
    out = _format_grading_material("A shell necklace", [])
    assert out == "Gold answer: A shell necklace"


def test_grading_material_rubric_only():
    out = _format_grading_material(None, ["mentions Flask", "mentions SQLite"])
    assert "Rubric (required points):" in out
    assert "- mentions Flask" in out
    assert "Gold answer" not in out


def test_grading_material_both():
    out = _format_grading_material("Flask app", ["mentions Flask"])
    assert "Gold answer: Flask app" in out
    assert "- mentions Flask" in out


def test_count_tokens_positive():
    assert count_tokens("hello world this is a test") > 0
