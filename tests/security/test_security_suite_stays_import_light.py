# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`workflow-trigger-lint.yml` runs `tests/security` on a four-package runner.

That job installs only pyyaml, pytest, pytest-xdist and vermin. It has no paths
filter, which is the point: it is the one job that still runs when a PR touches
nothing but workflow files. Keeping it that cheap is what lets it stay unfiltered.

So a file under `tests/security` may not import torch, transformers, numpy,
unsloth or unsloth_zoo at module scope unless the workflow explicitly ignores it.
Three files legitimately need those deps, and a torch-installing job runs them; the
workflow names them in `--ignore` so collection does not error.

The failure this guards against is quiet rather than loud. Under `-n 4` xdist does
not abort the session on a collection error, so an unlisted heavy import does not
turn the job red in an obvious way while its own tests stop running entirely. A
test that is silently never executed is worse than one that fails.

Pure AST and text: no imports of the modules under discussion, so this file is
itself safe to collect on the light runner.
"""

import ast
import pathlib
import re


HEAVY = {"torch", "transformers", "numpy", "unsloth", "unsloth_zoo", "peft", "trl"}

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_WORKFLOW = _REPO / ".github" / "workflows" / "workflow-trigger-lint.yml"


def _module_level_statements(body):
    """Statements that run at import time, including inside module-level control flow.

    `try: import torch / except ImportError: ...` is the ordinary way to write an
    optional dependency, and it executes during collection exactly like a bare import.
    Looking only at direct children of the module missed it, so a file written that way
    was reported as light, never added to the workflow's ignore list, and would stop
    collecting on the light runner - the failure this guard exists to prevent.

    Function bodies are still excluded, since an import in there is paid lazily. A
    CLASS body is not: it executes while the class is built, which happens at import
    time, so `class C: import torch` costs exactly as much as a bare import.
    """
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.ClassDef):
            yield from _module_level_statements(node.body)
            continue
        yield node
        for field in ("body", "orelse", "finalbody", "handlers"):
            for child in getattr(node, field, []) or []:
                if isinstance(child, ast.ExceptHandler):
                    yield from _module_level_statements(child.body)
                elif isinstance(child, ast.stmt):
                    yield from _module_level_statements([child])
        # `match` keeps its suites under `cases`, not under any of the fields above, so
        # a module-level `match ...: case _: import torch` walked straight past.
        for case in getattr(node, "cases", []) or []:
            yield from _module_level_statements(case.body)


def _module_level_heavy_imports(path):
    """Top-level import names only. An import inside a function is paid lazily."""
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    found = set()
    for node in _module_level_statements(tree.body):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # A relative import has no module of its own to blame.
            names = [node.module or ""] if node.level == 0 else []
        for name in names:
            root = name.split(".")[0]
            if root in HEAVY:
                found.add(root)
    return found


def _ignored_by_the_workflow():
    text = _WORKFLOW.read_text(encoding = "utf-8")
    return {
        pathlib.PurePosixPath(m).name
        for m in re.findall(r"--ignore=(tests/security/[\w./-]+\.py)", text)
    }


def test_the_workflow_still_runs_the_security_suite():
    """Guards the guard: if the step stops naming the suite, the rest is vacuous."""
    assert _WORKFLOW.exists(), _WORKFLOW
    assert "tests/security" in _WORKFLOW.read_text(encoding = "utf-8")


def test_heavy_imports_are_declared_to_the_light_runner():
    ignored = _ignored_by_the_workflow()
    offenders = {}
    for path in sorted(_HERE.glob("test_*.py")):
        heavy = _module_level_heavy_imports(path)
        if heavy and path.name not in ignored:
            offenders[path.name] = sorted(heavy)
    assert not offenders, (
        "these tests/security files import a runtime dependency at module scope but "
        f"are not in the workflow's --ignore list, so they will error during collection "
        f"on the four-package runner and then silently not run: {offenders}. Either "
        "move the import inside the test, or add --ignore for the file in "
        ".github/workflows/workflow-trigger-lint.yml."
    )


def test_the_ignore_list_has_no_stale_entries():
    """An ignore for a file that no longer needs it hides the file for no reason."""
    stale = []
    for name in sorted(_ignored_by_the_workflow()):
        path = _HERE / name
        if not path.exists():
            stale.append(f"{name} (no such file)")
        elif not _module_level_heavy_imports(path):
            stale.append(f"{name} (no longer imports anything heavy)")
    assert not stale, f"drop these from --ignore: {stale}"


def test_the_guard_is_not_vacuous():
    """The list must actually be non-empty, or the two tests above prove nothing."""
    ignored = _ignored_by_the_workflow()
    assert ignored, "the workflow names no --ignore, so this suite is not being checked"
    with_heavy = [
        path.name for path in _HERE.glob("test_*.py") if _module_level_heavy_imports(path)
    ]
    assert with_heavy, "no file imports a heavy dep, so the ignore list should be empty"
