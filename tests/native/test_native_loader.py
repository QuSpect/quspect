"""Tests for :mod:`quspect._native_loader` (scalar / AVX2 module selection)."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from quspect import _native_loader
from quspect._native_loader import (
    AVX2_MODULE,
    DISABLE_AVX2_ENV,
    SCALAR_MODULE,
    kernel_backend,
    load_core,
)


def _fake_core(name: str, *, backend: str, avx2: bool) -> SimpleNamespace:
    return SimpleNamespace(
        __name__=name,
        cpu_supports_avx2=lambda: avx2,
        get_runtime_capabilities=lambda: {
            "backend_f64": backend,
            "backend_f32": backend,
            "avx2_available": avx2,
        },
    )


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear the loader cache and the env override around every test."""
    monkeypatch.delenv(DISABLE_AVX2_ENV, raising=False)
    load_core.cache_clear()
    yield
    load_core.cache_clear()


def _install_importer(
    monkeypatch: pytest.MonkeyPatch,
    modules: dict[str, Any],
    errors: dict[str, ImportError] | None = None,
) -> list[str]:
    """Replace ``importlib.import_module`` inside the loader with a fake."""
    imported: list[str] = []
    failing = errors or {}

    def fake_import(name: str) -> Any:
        imported.append(name)
        if name in failing:
            raise failing[name]
        return modules[name]

    monkeypatch.setattr(_native_loader.importlib, "import_module", fake_import)
    return imported


def test_prefers_avx2_when_cpu_supports_it(monkeypatch: pytest.MonkeyPatch) -> None:
    scalar = _fake_core(SCALAR_MODULE, backend="scalar", avx2=True)
    avx2 = _fake_core(AVX2_MODULE, backend="avx2", avx2=True)
    imported = _install_importer(
        monkeypatch, {SCALAR_MODULE: scalar, AVX2_MODULE: avx2}
    )

    assert load_core() is avx2
    assert kernel_backend() == "avx2"
    assert imported == [SCALAR_MODULE, AVX2_MODULE]


def test_uses_scalar_when_cpu_lacks_avx2(monkeypatch: pytest.MonkeyPatch) -> None:
    scalar = _fake_core(SCALAR_MODULE, backend="scalar", avx2=False)
    imported = _install_importer(monkeypatch, {SCALAR_MODULE: scalar})

    assert load_core() is scalar
    assert kernel_backend() == "scalar"
    assert AVX2_MODULE not in imported


def test_env_var_forces_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DISABLE_AVX2_ENV, "1")
    scalar = _fake_core(SCALAR_MODULE, backend="scalar", avx2=True)
    avx2 = _fake_core(AVX2_MODULE, backend="avx2", avx2=True)
    imported = _install_importer(
        monkeypatch, {SCALAR_MODULE: scalar, AVX2_MODULE: avx2}
    )

    assert load_core() is scalar
    assert kernel_backend() == "scalar"
    assert imported == [SCALAR_MODULE]


@pytest.mark.parametrize("value", ["0", "", "true"])
def test_env_var_other_values_do_not_disable(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(DISABLE_AVX2_ENV, value)
    scalar = _fake_core(SCALAR_MODULE, backend="scalar", avx2=True)
    avx2 = _fake_core(AVX2_MODULE, backend="avx2", avx2=True)
    _install_importer(monkeypatch, {SCALAR_MODULE: scalar, AVX2_MODULE: avx2})

    assert load_core() is avx2


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError(f"No module named {AVX2_MODULE!r}", name=AVX2_MODULE),
        ImportError("dlopen failed: undefined symbol"),
    ],
)
def test_falls_back_to_scalar_when_avx2_import_fails(
    monkeypatch: pytest.MonkeyPatch, error: ImportError
) -> None:
    scalar = _fake_core(SCALAR_MODULE, backend="scalar", avx2=True)
    imported = _install_importer(
        monkeypatch, {SCALAR_MODULE: scalar}, {AVX2_MODULE: error}
    )

    assert load_core() is scalar
    assert kernel_backend() == "scalar"
    assert imported == [SCALAR_MODULE, AVX2_MODULE]


def test_returns_none_when_scalar_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ModuleNotFoundError(
        f"No module named {SCALAR_MODULE!r}", name=SCALAR_MODULE
    )
    imported = _install_importer(monkeypatch, {}, {SCALAR_MODULE: error})

    assert load_core() is None
    assert kernel_backend() is None
    assert imported == [SCALAR_MODULE]


def test_reraises_unrelated_module_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    error = ModuleNotFoundError("No module named 'numpy'", name="numpy")
    _install_importer(monkeypatch, {}, {SCALAR_MODULE: error})

    with pytest.raises(ModuleNotFoundError, match="numpy"):
        load_core()


def test_reraises_broken_scalar_module(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_importer(monkeypatch, {}, {SCALAR_MODULE: ImportError("broken extension")})

    with pytest.raises(ImportError, match="broken extension"):
        load_core()


def test_result_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    scalar = _fake_core(SCALAR_MODULE, backend="scalar", avx2=False)
    imported = _install_importer(monkeypatch, {SCALAR_MODULE: scalar})

    assert load_core() is load_core()
    assert imported == [SCALAR_MODULE]


def test_real_extension_reports_consistent_backend() -> None:
    """With the real build, the loaded module and its backend must agree."""
    core = load_core()
    if core is None:
        pytest.skip("QuSpect native extension not built")
    assert isinstance(core, ModuleType)
    backend = kernel_backend()
    assert backend in ("scalar", "avx2")
    assert core.__name__ == f"quspect._core_{backend}"
    caps = core.get_runtime_capabilities()
    assert caps["backend_f32"] == backend
    scalar = importlib.import_module(SCALAR_MODULE)
    assert bool(scalar.cpu_supports_avx2()) == bool(caps["avx2_available"])
    assert scalar.get_runtime_capabilities()["backend_f64"] == "scalar"


def test_load_core_is_public_in_quspect_native() -> None:
    """``load_core`` is a supported public API of :mod:`quspect.native`."""
    import quspect.native as native
    from quspect.native import load_core as public_load_core

    assert "load_core" in native.__all__
    assert public_load_core is load_core


def test_quspect_native_all_names_resolve() -> None:
    """Every name in ``quspect.native.__all__`` exists on the module."""
    import quspect.native as native

    for name in native.__all__:
        assert hasattr(native, name), name
    for name in (
        "ExecutionResult",
        "NativeExecutor",
        "NativeImportError",
        "capabilities",
        "decode_qsal",
        "get_num_threads",
        "set_num_threads",
        "validate",
    ):
        assert name in native.__all__, name
