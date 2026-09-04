from pathlib import Path

import pairedrl

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_package_imports():
    assert pairedrl.__version__ == "0.0.1"


def test_preregistration_lists_six_hypotheses():
    text = (DOCS / "PREREGISTRATION.md").read_text(encoding="utf-8")
    for tag in ["H1", "H2", "H3", "H4", "H5", "H6"]:
        assert f"### {tag}" in text, f"missing heading for {tag}"


def test_docs_contain_no_dashes_of_the_long_kind():
    em_dash, en_dash = "\u2014", "\u2013"
    for name in ["PREREGISTRATION.md", "DAILY_LOG.md", "DEVIATIONS.md", "RUN_LEDGER.md"]:
        text = (DOCS / name).read_text(encoding="utf-8")
        assert em_dash not in text, f"em dash in {name}"
        assert en_dash not in text, f"en dash in {name}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert em_dash not in readme and en_dash not in readme
