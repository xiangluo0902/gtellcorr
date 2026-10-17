"""Write the Python script of a notebook, in the "percent" cell format.

Code cells become code under ``# %%``, Markdown cells comments under
``# %% [markdown]`` (the format VS Code, Spyder and jupytext read as cells); the result
is passed through ``ruff format`` (a development dependency). The paper notebooks in
``examples/paper/`` are kept with their scripts this way.

    python tools/dev/notebook_to_script.py NOTEBOOK.ipynb [...]          # write NOTEBOOK.py
    python tools/dev/notebook_to_script.py --check NOTEBOOK.ipynb [...]  # fail if out of date
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def script_of(notebook: Path) -> str:
    """The percent-format script of *notebook*, formatted by ruff."""
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    parts = [f'"""Script version of {notebook.name}; the notebook explains every step."""']
    for cell in cells:
        source = "".join(cell["source"]).rstrip()
        if cell["cell_type"] == "markdown":
            body = "\n".join(f"# {line}".rstrip() for line in source.splitlines())
            parts.append(f"# %% [markdown]\n{body}")
        elif cell["cell_type"] == "code":
            parts.append(f"# %%\n{source}")
    text = "\n\n".join(parts) + "\n"
    done = subprocess.run(
        ["ruff", "format", "--stdin-filename", notebook.with_suffix(".py").name, "-"],
        input=text, capture_output=True, text=True, check=True,
    )  # fmt: skip
    return done.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="only check that scripts match")
    args = parser.parse_args()
    stale = []
    for notebook in args.notebooks:
        script = notebook.with_suffix(".py")
        text = script_of(notebook)
        if args.check:
            if not script.exists() or script.read_text(encoding="utf-8") != text:
                stale.append(script)
        else:
            script.write_text(text, encoding="utf-8")
            print(f"wrote {script}")
    if stale:
        sys.exit("out of date: " + ", ".join(str(s) for s in stale))


if __name__ == "__main__":
    main()
