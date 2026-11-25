"""Start a new project from templates/project.

    python new_project.py p9_qkd_rl "Can RL tune BB84 under drifting channel noise?"

Creates p9_qkd_rl/run.py (a working multi-seed experiment skeleton) and
p9_qkd_rl/README.md, with the name and question filled in.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "templates" / "project"


def main():
    ap = argparse.ArgumentParser(description="Create a new project folder from the template.")
    ap.add_argument("name", help="folder name, e.g. p9_qkd_rl (lowercase letters, digits, underscores)")
    ap.add_argument("question", help="the project's one-line research question")
    a = ap.parse_args()

    if not re.fullmatch(r"[a-z][a-z0-9_]*", a.name):
        sys.exit("Name must start with a letter and use only lowercase letters, digits and underscores.")
    dest = ROOT / a.name
    if dest.exists():
        sys.exit(f"{a.name}/ already exists; pick another name.")

    created = []
    for src in sorted(TEMPLATE.rglob("*")):
        if src.is_dir() or "__pycache__" in src.parts:
            continue
        out = dest / src.relative_to(TEMPLATE)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8")
        out.write_text(text.replace("{{NAME}}", a.name).replace("{{QUESTION}}", a.question), encoding="utf-8")
        created.append(out.relative_to(ROOT).as_posix())

    print("Created:")
    for path in created:
        print(f"  {path}")
    print(f"\nTry it:  python {a.name}/run.py --seeds 3")
    print(f"Then replace run_once() in {a.name}/run.py with your experiment.")


if __name__ == "__main__":
    main()
