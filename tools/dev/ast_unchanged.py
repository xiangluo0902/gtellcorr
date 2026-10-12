#!/usr/bin/env python
"""Check that edits to Python files changed only docstrings, comments and layout.

    python tools/dev/ast_unchanged.py [--ref HEAD] [PATH ...]

For every Python file under the given paths (default: ``src``) that differs from
git revision ``--ref``, both versions are parsed, every bare string statement is
removed (module, class and function docstrings, and the attribute docstrings that
follow a field), and the remaining syntax trees are compared. Comments and
formatting never reach the syntax tree, so a documentation-only edit leaves it
unchanged. Files whose code changed are listed and the exit status is 1.

New files (absent at ``--ref``) and deleted files are reported separately; they
are code changes by definition.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path


class _StripDocstrings(ast.NodeTransformer):
    """Drop every bare string statement: docstrings and attribute docstrings.

    A string literal used as a statement has no effect when the code runs, so it is
    documentation wherever it appears.
    """

    def generic_visit(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        for name in ("body", "orelse", "finalbody"):
            body = getattr(node, name, None)
            if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
                kept = [stmt for stmt in body if not _is_string_statement(stmt)]
                setattr(node, name, kept or ([ast.Pass()] if name == "body" else []))
        return node


def _is_string_statement(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def code_signature(source: str) -> str:
    """The syntax tree of *source* without docstrings, as a comparable string."""
    tree = _StripDocstrings().visit(ast.parse(source))
    return ast.dump(tree, include_attributes=False)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=["src"])
    parser.add_argument("--ref", default="HEAD")
    args = parser.parse_args(argv)

    changed = _git("diff", "--name-status", args.ref, "--", *args.paths).splitlines()
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *args.paths).splitlines()
    code_changed, added, removed, checked = [], [], [], 0
    for line in changed + [f"A\t{p}" for p in untracked]:
        status, *names = line.split("\t")
        path = names[-1]
        if not path.endswith(".py"):
            continue
        if status.startswith("A"):
            added.append(path)
            continue
        if status.startswith("D"):
            removed.append(path)
            continue
        old_path = names[0]
        before = _git("show", f"{args.ref}:{old_path}")
        after = Path(path).read_text(encoding="utf-8")
        checked += 1
        if code_signature(before) != code_signature(after):
            code_changed.append(path)

    print(f"{checked} modified Python file(s) checked against {args.ref}")
    for label, paths in (("new", added), ("deleted", removed), ("CODE CHANGED", code_changed)):
        for path in paths:
            print(f"  {label}: {path}")
    if not code_changed:
        print("only docstrings, comments and layout changed")
    return 1 if code_changed else 0


if __name__ == "__main__":
    sys.exit(main())
