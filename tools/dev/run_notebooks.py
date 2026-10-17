"""Execute notebooks in place and check that their outputs name no local path.

The manual (``docs/manual``) and the paper notebooks (``examples/paper``) are committed
with their outputs. Each notebook runs in its own folder; a failing cell stops that
notebook (its outputs up to the failure are kept) and the others still run. Outputs that
contain the home directory are reported, since they would publish local paths.

    python tools/dev/run_notebooks.py NOTEBOOK.ipynb [...]

Needs ``nbclient`` and ``ipykernel`` in the environment that runs gtellcorr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nbclient
import nbformat


def run(path: Path) -> str:
    """Execute *path* in place; return a one-line status."""
    started = time.time()
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}
    )
    try:
        client.execute()
        status = "ok"
    except nbclient.exceptions.CellExecutionError as exc:
        status = "FAILED: " + str(exc).strip().splitlines()[-1][:300]
    nbformat.write(nb, path)
    return f"{path}: {status} in {(time.time() - started) / 60:.1f} min"


#: Environment variables whose folders the notebooks read and write.
DATA_VARIABLES = ("GTELLCORR_EXAMPLE", "GTELLCORR_PAPER_DATA", "GTELLCORR_PAPER_OUT")


def local_paths(path: Path) -> list[str]:
    """Outputs of *path* that contain the home directory or a data folder."""
    roots = {str(Path.home())}
    for name in DATA_VARIABLES:
        if os.environ.get(name):
            roots.add(str(Path(os.environ[name]).resolve()))
    found = []
    for i, cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
        text = json.dumps([o for o in cell.get("outputs", [])])
        if any(root in text for root in roots):
            found.append(f"cell {i}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="+", type=Path)
    args = parser.parse_args()
    bad = False
    for path in args.notebooks:
        line = run(path)
        print(line, flush=True)
        leaks = local_paths(path)
        if leaks:
            print(f"  local paths in the outputs of {', '.join(leaks)}", flush=True)
        bad |= "FAILED" in line or bool(leaks)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
