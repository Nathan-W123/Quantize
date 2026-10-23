"""Copy benchmark results out of output/ and into version control.

output/ is gitignored, which is right for scratch but wrong for these: one
accuracy-upgrades sweep is hours of CPU, and losing it to a cleared working
directory means re-running it rather than reading it. Twice during this
project a result set was lost exactly that way, and the second time a
five-hour run was abandoned because its odds of surviving were poor.

This copies the JSON into reports/accuracy_upgrades/, which IS tracked, and
writes a small index alongside recording the code vintage of every row. Commit
the result; a number that cost hours deserves to be as durable as the code
that produced it.

    python scripts/archive_benchmark_results.py            # archive everything
    python scripts/archive_benchmark_results.py hf_6-31g   # one tag
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "output"
_DEST = _ROOT / "reports" / "accuracy_upgrades"


def archive(tag_filter: str | None = None) -> list[Path]:
    _DEST.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index: dict[str, dict] = {}

    for src in sorted(_SRC.glob("accuracy_upgrades_*.json")):
        tag = src.stem.replace("accuracy_upgrades_", "")
        if tag_filter and tag != tag_filter:
            continue
        data = json.loads(src.read_text(encoding="utf-8"))

        vintages: Counter = Counter()
        configs: Counter = Counter()
        for rec in data.values():
            for cfg, entry in rec.items():
                configs[cfg] += 1
                if isinstance(entry, dict):
                    vintages[entry.get("code_vintage") or "unstamped"] += 1

        dest = _DEST / src.name
        shutil.copy2(src, dest)
        written.append(dest)
        index[tag] = {
            "molecules": sorted(data),
            "configs": dict(sorted(configs.items())),
            "code_vintages": dict(sorted(vintages.items())),
            "mixed_vintage": len(vintages) > 1,
        }
        mixed = " MIXED VINTAGE" if len(vintages) > 1 else ""
        print(f"  {tag:<24} {len(data):>2} molecules, "
              f"{sum(configs.values()):>3} rows, "
              f"{len(vintages)} vintage(s){mixed}")

    if index:
        path = _DEST / "index.json"
        existing = (json.loads(path.read_text(encoding="utf-8"))
                    if path.exists() else {})
        existing.update(index)
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        written.append(path)
        print(f"  index -> {path.relative_to(_ROOT)}")
    else:
        print("  nothing to archive")
    return written


if __name__ == "__main__":
    archive(sys.argv[1] if len(sys.argv) > 1 else None)
