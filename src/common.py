"""Shared paths and tiny helpers used by every pipeline script.

All scripts can be run from any working directory because every path is anchored to the repository root.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
POLICIES = ROOT / "data" / "policies"
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"
SQL_FILE = ROOT / "sql" / "queries.sql"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
STATE = ROOT / "state"

for _d in (CLEAN, OUTPUTS, REPORTS, STATE, ROOT / "sql"):
    _d.mkdir(parents=True, exist_ok=True)

# mock_llm.py lives at the repository root and must stay unmodified.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Report:
    """Print every line and also keep it so it can be saved under reports/ (plain text, no images)."""

    def __init__(self, name):
        self.name, self.lines = name, []

    def __call__(self, text=""):
        text = str(text)
        print(text)
        self.lines.append(text)

    def save(self):
        (REPORTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n", encoding="utf-8")
