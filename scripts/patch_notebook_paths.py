#!/usr/bin/env python3
"""Prepend RESULTS_DIR setup and replace literal /data/results paths in notebooks."""
from __future__ import annotations

import json
import re
from pathlib import Path

SETUP = """import os
from pathlib import Path

_cwd = Path.cwd().resolve()
if os.environ.get("RESULTS_DIR"):
    RESULTS_DIR = Path(os.environ["RESULTS_DIR"]).expanduser().resolve()
elif (_cwd / "data").is_dir():
    RESULTS_DIR = (_cwd / "data" / "results").resolve()
else:
    RESULTS_DIR = (_cwd.parent / "data" / "results").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

"""


def split_lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return [line + "\n" for line in text.split("\n")[:-1]]


def join_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def replace_paths(text: str) -> str:
    text = text.replace(
        "RESULTS = '/data/results/nlp_comparison.csv'",
        "RESULTS = str(RESULTS_DIR / 'nlp_comparison.csv')",
    )
    return re.sub(r"'/data/results/([^']*)'", r"str(RESULTS_DIR / '\1')", text)


def patch_notebooks(repo_root: Path) -> None:
    for path in sorted((repo_root / "notebooks").glob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        hit_indices = [
            i
            for i, c in enumerate(nb.get("cells", []))
            if c.get("cell_type") == "code" and "/data/results" in join_source(c)
        ]
        if not hit_indices:
            continue

        prepended = False
        for idx in hit_indices:
            cell = nb["cells"][idx]
            updated = replace_paths(join_source(cell))
            if not prepended and "RESULTS_DIR.mkdir" not in updated:
                updated = SETUP + updated
                prepended = True
            cell["source"] = split_lines(updated)

        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path.relative_to(repo_root))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    patch_notebooks(root)
