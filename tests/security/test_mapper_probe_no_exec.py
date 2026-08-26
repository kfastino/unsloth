# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`_get_new_mapper` used to `exec` an HTTPS response body.

When `from_pretrained` is handed a model name the installed `mapper.py` does not know,
`loader_utils` fetches mapper.py from raw.githubusercontent.com and used to run
`exec(response.text, namespace)`. That is arbitrary code execution inside an ordinary
model load, gated only on transport integrity: nothing checked that the body was the
file it claimed to be, and `exec` into a fresh dict still gets full builtins.

Only one thing in that file is data. The probe now `ast.literal_eval`s the
`__INT_TO_FLOAT_MAPPER` dict literal out of the fetched text and derives the five
tables with `mapper.build_mappers`, the installed version's own code. A hostile body
can therefore change what the probe *reports*, which was always true, but can no
longer run.

CPU-only and network-free: `requests.get` is stubbed, and `tests/security/conftest.py`
blocks non-loopback sockets anyway.
"""

import builtins
import types
from contextlib import contextmanager

import pytest

from unsloth.models import loader_utils
from unsloth.models.mapper import build_mappers


REAL_MAPPER = open(
    __import__("pathlib").Path(loader_utils.__file__).with_name("mapper.py"),
    encoding = "utf-8",
).read()


class _Response:
    """The streaming half of `requests.Response`, which is all the probe uses.

    The probe reads in chunks and stops at its byte cap while reading, because
    `requests.get` would otherwise buffer and decode the whole body before any length
    check could run. It also follows redirects itself, since `requests` drains each
    intermediate body inside `get`, so a status and headers are needed as well.
    A fake offering only `.text` would let either of those regress silently.
    """

    def __init__(
        self,
        text,
        status_code = 200,
        headers = None,
    ):
        self.encoding = "utf-8"
        self.status_code = status_code
        self.headers = headers or {}
        self._body = text.encode("utf-8")

    def iter_content(self, chunk_size = 1):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False


@contextmanager
def _serving(body, monkeypatch):
    """Makes the probe's `requests.get` return `body`."""
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response(body))
    if not hasattr(requests, "compat"):
        monkeypatch.setattr(requests, "compat", types.SimpleNamespace(urljoin = lambda b, u: u))
    yield


@pytest.fixture
def no_dynamic_execution(monkeypatch):
    """Records any exec/eval reached during the probe.

    It RECORDS rather than raises. `_get_new_mapper` catches every exception, so an
    AssertionError raised here is swallowed and the probe returns its five empty
    tables - which then satisfies a `len(result) == 5` assertion just as well as the
    fixed implementation does. A tripwire the caller can eat is not a tripwire, so the
    test asserts on this list instead.
    """
    calls = []

    def _forbidden(name):
        def _record(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"_get_new_mapper called {name}()")

        return _record

    monkeypatch.setattr(builtins, "exec", _forbidden("exec"))
    monkeypatch.setattr(builtins, "eval", _forbidden("eval"))
    return calls


# --- the payload cannot run --------------------------------------------------

MARKER = "/tmp/unsloth_mapper_probe_marker"

PAYLOADS = [
    f"import os; os.system('touch {MARKER}')\n__INT_TO_FLOAT_MAPPER = {{}}\n",
    f"__INT_TO_FLOAT_MAPPER = {{}}\nimport os\nos.system('touch {MARKER}')\n",
    f"open({MARKER!r}, 'w').close()\n__INT_TO_FLOAT_MAPPER = {{}}\n",
    "__INT_TO_FLOAT_MAPPER = __import__('os').environ\n",
    "class X:\n    def __init__(self): __import__('os').getpid()\n__INT_TO_FLOAT_MAPPER = X()\n",
]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_payload_in_the_response_is_never_executed(payload, monkeypatch, tmp_path):
    import os

    if os.path.exists(MARKER):
        os.remove(MARKER)

    with _serving(payload, monkeypatch):
        result = loader_utils._get_new_mapper()

    assert not os.path.exists(MARKER), "the fetched body executed"
    # Whatever it returns, it must be the five-table shape and carry nothing useful.
    assert len(result) == 5
    assert all(isinstance(table, dict) for table in result)


def test_probe_does_not_call_exec_or_eval(no_dynamic_execution, monkeypatch):
    """Stronger than the marker: the builtins are not reached at all.

    The assertion is on the recorded calls. Asserting only the returned shape would
    also hold for the pre-change implementation, since the AssertionError the fixture
    raises is caught by the probe's own bare except.
    """
    with _serving(REAL_MAPPER, monkeypatch):
        result = loader_utils._get_new_mapper()
    assert no_dynamic_execution == [], f"the probe reached {no_dynamic_execution}"
    # And it still did the work, rather than falling into the except and returning
    # empties, which is the other way this test could pass for the wrong reason.
    assert len(result) == 5
    assert all(result[:3]), "the probe returned nothing, so it proved nothing"


def test_a_body_that_is_not_python_is_survivable(monkeypatch):
    for body in ("", "<html>404</html>", "\x00\x01\x02", "def "):
        with _serving(body, monkeypatch):
            assert loader_utils._get_new_mapper() == ({}, {}, {}, {}, {})


def test_a_body_without_the_source_table_returns_nothing(monkeypatch):
    with _serving("SOMETHING_ELSE = {'a': 1}\n", monkeypatch):
        assert loader_utils._get_new_mapper() == ({}, {}, {}, {}, {})


def test_a_deeply_nested_body_is_survivable(monkeypatch):
    """literal_eval evaluates literals only, but it is not DoS-safe.

    ast.parse builds the whole tree before any literal-only check runs, so nesting past
    the compiler's recursion limit raises RecursionError, which is not a ValueError.
    The probe's bare except catches it; this pins that it stays caught.
    """
    body = "__INT_TO_FLOAT_MAPPER = " + "[" * 20000 + "]" * 20000
    with _serving(body, monkeypatch):
        assert loader_utils._get_new_mapper() == ({}, {}, {}, {}, {})


def test_an_oversized_body_is_not_parsed_at_all(monkeypatch):
    """The size cap, which bounds parse cost rather than correctness.

    `requests`' timeout is per-read, not total, so a body can be arbitrarily large. The
    real mapper.py is around 50KB against a 10MB cap.
    """
    body = "__INT_TO_FLOAT_MAPPER = {'a' : ('b',)}\n" + ("# padding\n" * 1_100_000)
    assert len(body) > 10_000_000
    with _serving(body, monkeypatch):
        assert loader_utils._get_new_mapper() == ({}, {}, {}, {}, {})


def test_the_real_mapper_is_far_below_the_cap():
    assert len(REAL_MAPPER) < 10_000_000 / 10


# --- the probe still works ---------------------------------------------------


def test_real_mapper_body_reproduces_the_installed_tables(monkeypatch):
    """Serving the installed mapper.py back must reproduce the installed tables."""
    with _serving(REAL_MAPPER, monkeypatch):
        result = loader_utils._get_new_mapper()

    from unsloth.models import mapper

    expected = (
        mapper.INT_TO_FLOAT_MAPPER,
        mapper.FLOAT_TO_INT_MAPPER,
        mapper.MAP_TO_UNSLOTH_16bit,
        mapper.FLOAT_TO_FP8_BLOCK_MAPPER,
        mapper.FLOAT_TO_FP8_ROW_MAPPER,
    )
    assert result == expected
    assert len(result[0]) > 100, "the probe returned an empty table"


def test_a_newer_table_is_picked_up(monkeypatch):
    """The probe's actual job: report a mapping the installed tables do not have."""
    body = (
        "__INT_TO_FLOAT_MAPPER = {\n"
        "    'unsloth/some-new-model-bnb-4bit': ('unsloth/some-new-model',),\n"
        "}\n"
    )
    with _serving(body, monkeypatch):
        int_to_float, float_to_int, _, _, _ = loader_utils._get_new_mapper()

    assert int_to_float["unsloth/some-new-model-bnb-4bit"] == "unsloth/some-new-model"
    assert float_to_int["unsloth/some-new-model"] == "unsloth/some-new-model-bnb-4bit"
