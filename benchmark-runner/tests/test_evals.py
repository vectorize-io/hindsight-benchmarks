import json

from hindsight_benchmark.evals import cost_usd, parse_pricing, tally

OPERATIONS = {
    "retain": ["evals/test_04_retain_fidelity.py"],
    "reflect": ["evals/test_02_reflect_answers.py"],
}

JUNIT = """<?xml version="1.0"?><testsuites><testsuite>
<testcase classname="evals.test_04_retain_fidelity" name="a"/>
<testcase classname="evals.test_04_retain_fidelity" name="b"><failure message="x"/></testcase>
<testcase classname="evals.test_02_reflect_answers" name="c"><error message="x"/></testcase>
<testcase classname="evals.test_02_reflect_answers" name="d"><skipped/></testcase>
<testcase classname="evals.test_02_reflect_answers" name="e"/>
</testsuite></testsuites>"""


def test_tally_and_cost(tmp_path):
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT)
    tokens = tmp_path / "tokens.jsonl"
    rows = [
        {"nodeid": "evals/test_04_retain_fidelity.py::test_x[a]", "calls": 2, "input_tokens": 1000, "cached_tokens": 400, "output_tokens": 100},
        {"nodeid": "evals/test_04_retain_fidelity.py::test_x[b]", "calls": 1, "input_tokens": 500, "cached_tokens": 0, "output_tokens": 50},
    ]
    tokens.write_text("\n".join(json.dumps(r) for r in rows))

    ops = tally([junit], [tokens], OPERATIONS)

    assert (ops["retain"]["passed"], ops["retain"]["failed"], ops["retain"]["score"]) == (1, 1, 50.0)
    # skipped is ignored; an error counts against the model
    assert (ops["reflect"]["total"], ops["reflect"]["errors"], ops["reflect"]["score"]) == (2, 1, 50.0)
    assert ops["retain"]["tokens"] == {"calls": 3, "input_tokens": 1500, "cached_tokens": 400, "output_tokens": 150}
    assert ops["reflect"]["tokens"]["calls"] == 0

    pricing = {"input": 1e-6, "cached_input": 1e-7, "output": 4e-6}
    # 1100 uncached * 1e-6 + 400 * 1e-7 + 150 * 4e-6
    assert cost_usd(ops["retain"]["tokens"], pricing) == round(0.0011 + 0.00004 + 0.0006, 4)
    assert cost_usd(ops["retain"]["tokens"], {}) is None


def test_parse_pricing_both_shapes():
    assert parse_pricing({"input_per_million": "0.1", "output_per_million": "0.95"}) == {
        "input": 0.1 / 1e6, "cached_input": 0.1 / 1e6, "output": 0.95 / 1e6}
    assert parse_pricing({"prompt": "0.000001", "completion": "0.000004", "input_cache_read": "0.0000001"}) == {
        "input": 1e-6, "output": 4e-6, "cached_input": 1e-7}


def test_parse_collected_and_readme():
    from hindsight_benchmark.evals import parse_collected, parse_readme_table

    lines = """<Dir hindsight-system-evals>
  <Package evals>
    <Package refresh-pages>
      <Module test_refresh_pages.py>
        <Coroutine test_refresh_pages[update/du-oncall-departed]>
        <Coroutine test_refresh_pages[update/du-renamed-service]>
        <Coroutine test_refresh_pages_benchmark_cost>
    <Package retain>
      <Module test_retain.py>
        <Coroutine test_retain_language[language/ja-personal-note]>""".splitlines()
    assert parse_collected(lines) == {
        "refresh_pages": {"update": ["du-oncall-departed", "du-renamed-service"], "benchmarks": ["test_refresh_pages_benchmark_cost"]},
        "retain": {"language": ["ja-personal-note"]},
    }
    readme = "| operation | category | cases | what it checks |\n| retain | language | 7 | keeps its language |\n"
    assert parse_readme_table(readme) == {("retain", "language"): "keeps its language"}


def test_summarize():
    from hindsight_benchmark.evals import summarize
    ops = {"a": {"score": 80.0, "cost_usd": 0.2}, "b": {"score": 60.0, "cost_usd": 0.1}, "c": {"score": None, "cost_usd": None}}
    assert summarize(ops) == {"score": 70.0, "cost_usd": 0.3}
