"""tools/dev/ast_unchanged.py: documentation-only edits keep the code signature."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "dev" / "ast_unchanged.py"
spec = importlib.util.spec_from_file_location("ast_unchanged", TOOL)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules["ast_unchanged"] = module
spec.loader.exec_module(module)
sig = module.code_signature

BEFORE = '''
"""Module."""
from dataclasses import dataclass

@dataclass
class A:
    x: int = 1
    """Old field doc."""

    def f(self, y):
        # a comment
        if y:
            return y + self.x
        return 0
'''


def test_documentation_edits_keep_the_signature() -> None:
    after = (
        BEFORE.replace('"""Module."""', '"""A new module docstring."""')
        .replace('"""Old field doc."""', '"""New field doc,\n    over two lines."""')
        .replace("# a comment", "# another comment")
        .replace("        if y:", '        """Added method docstring."""\n        if y:')
    )
    assert sig(after) == sig(BEFORE)


def test_code_edits_change_the_signature() -> None:
    assert sig(BEFORE.replace("x: int = 1", "x: int = 2")) != sig(BEFORE)
    assert sig(BEFORE.replace("return y + self.x", "return y - self.x")) != sig(BEFORE)
    assert sig(BEFORE.replace("return 0", "return None")) != sig(BEFORE)
