"""Cross-validate ``quspect.gates`` (Gate, templates, gate_matrix) with qiskit.

``gate_matrix`` is big-endian (first qubit argument = most significant bit),
so it is compared with ``Operator(qiskit_gate).reverse_qargs().data``.
``parameterized.py`` exposes gate *templates* (``RX``, ``CU3`` ...) that are
turned into concrete gates with ``Gate.with_params``; there are no other
factory functions in that module.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import pytest
import qiskit.circuit.library as qlib

from qiskit_tests.helpers import GATE_SPECS, assert_unitary_equal, qiskit_gate_matrix_be
from quspect.gates import ALL_PARAMETERIZED, ALL_STANDARD, Gate, gate_matrix
from quspect.gates import parameterized as pz
from quspect.gates import standard as st

_EXPECTED_STANDARD = {
    "sx": 1,
    "sxdg": 1,
    "id": 1,
    "h": 1,
    "x": 1,
    "y": 1,
    "z": 1,
    "s": 1,
    "sdg": 1,
    "t": 1,
    "tdg": 1,
    "cx": 2,
    "cz": 2,
    "swap": 2,
    "ccx": 3,
}
_EXPECTED_PARAM = {
    "rx": (1, 1),
    "ry": (1, 1),
    "rz": (1, 1),
    "p": (1, 1),
    "u1": (1, 1),
    "u2": (1, 2),
    "u3": (1, 3),
    "crx": (2, 1),
    "cry": (2, 1),
    "crz": (2, 1),
    "cp": (2, 1),
    "cu3": (2, 3),
}
_ANGLES = [0.0, math.pi, -math.pi, math.pi / 2, 2 * math.pi, 1e-9, 0.731, -2.4, 7.5]


def test_all_standard_contents() -> None:
    """ALL_STANDARD lists the 15 fixed gates with correct arity, no params."""
    assert {g.name: g.num_qubits for g in ALL_STANDARD} == _EXPECTED_STANDARD
    assert all(g.params == () for g in ALL_STANDARD)
    assert st.I.name == "id" and st.SDG.label == "S†"


def test_all_parameterized_templates() -> None:
    """ALL_PARAMETERIZED lists the 12 templates, none carrying parameters."""
    assert {g.name: g.num_qubits for g in ALL_PARAMETERIZED} == {
        k: v[0] for k, v in _EXPECTED_PARAM.items()
    }
    assert all(g.params == () for g in ALL_PARAMETERIZED)
    assert pz.CU3 in ALL_PARAMETERIZED


def test_gate_tables_cover_qiskit_specs() -> None:
    """Every quspect gate has a qiskit counterpart in GATE_SPECS."""
    names = {g.name for g in ALL_STANDARD} | {g.name for g in ALL_PARAMETERIZED}
    assert names == set(GATE_SPECS)
    for name, (nq, npar) in _EXPECTED_PARAM.items():
        assert GATE_SPECS[name].num_qubits == nq
        assert GATE_SPECS[name].num_params == npar


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("gate", ALL_STANDARD, ids=lambda g: g.name)
def test_standard_gate_matrix_matches_qiskit(gate: Gate) -> None:
    """Fixed gate matrices equal qiskit's (big-endian) exactly."""
    ref = qiskit_gate_matrix_be(GATE_SPECS[gate.name].qiskit_gate())
    mat = gate_matrix(gate)
    assert mat.dtype == np.complex128
    assert_unitary_equal(mat, ref, atol=1e-15)


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("template", ALL_PARAMETERIZED, ids=lambda g: g.name)
def test_template_with_params_matrix_matches_qiskit(template: Gate) -> None:
    """``template.with_params(...)`` matrices equal qiskit at many angles."""
    spec = GATE_SPECS[template.name]
    rng = np.random.default_rng(zlib.crc32(template.name.encode()))
    for _ in range(8):
        params = [
            float(_ANGLES[int(rng.integers(len(_ANGLES)))])
            if rng.random() < 0.5
            else float(rng.uniform(-10, 10))
            for _ in range(spec.num_params)
        ]
        gate = template.with_params(*params)
        assert gate.name == template.name
        assert gate.label == template.label
        mat = gate_matrix(gate)
        assert_unitary_equal(mat, qiskit_gate_matrix_be(spec.qiskit_gate(*params)))
        assert_unitary_equal(mat.conj().T @ mat, np.eye(mat.shape[0]))


@pytest.mark.qiskit_ref
def test_u_family_relations_match_qiskit() -> None:
    """u1 == p, u2(phi, lam) == u3(pi/2, phi, lam), cu3 == qiskit CU(gamma=0)."""
    for ang in _ANGLES:
        assert_unitary_equal(
            gate_matrix(pz.U1.with_params(ang)), gate_matrix(pz.P.with_params(ang))
        )
        assert_unitary_equal(
            gate_matrix(pz.U2.with_params(ang, -ang / 3)),
            gate_matrix(pz.U3.with_params(math.pi / 2, ang, -ang / 3)),
        )
        assert_unitary_equal(
            gate_matrix(pz.CU3.with_params(ang, 0.3, -0.2)),
            qiskit_gate_matrix_be(qlib.CUGate(ang, 0.3, -0.2, 0.0)),
        )


def test_gate_matrix_returns_fresh_copy() -> None:
    """Mutating a returned matrix does not affect later calls."""
    m = gate_matrix(st.H)
    m[:] = 0
    assert np.abs(gate_matrix(st.H)).max() > 0.5


def test_gate_value_semantics() -> None:
    """Gate is frozen, hashable, compares without label; repr shows params."""
    g1 = Gate(name="rx", num_qubits=1, params=(0.5,), label="A")
    g2 = Gate(name="rx", num_qubits=1, params=(0.5,), label="B")
    assert g1 == g2
    assert hash(g1) == hash(g2)
    assert g1 != pz.RX.with_params(0.6)
    with pytest.raises(AttributeError):
        g1.name = "ry"  # type: ignore[misc]
    assert repr(g1) == "Gate(name='rx', num_qubits=1, params=(0.5,), label='A')"
    assert repr(st.H) == "Gate(name='h', num_qubits=1, label='H')"
    assert repr(Gate(name="x", num_qubits=1)) == "Gate(name='x', num_qubits=1)"
    # Parameter count does not have to equal the qubit count.
    assert Gate(name="u3", num_qubits=1, params=(1.0, 2.0, 3.0)).params == (
        1.0,
        2.0,
        3.0,
    )


@pytest.mark.parametrize(
    ("build", "exc"),
    [
        (lambda: Gate(name="x", num_qubits=0), ValueError),
        (lambda: pz.RX.with_params(), ValueError),
        (lambda: gate_matrix(Gate(name="nope", num_qubits=1)), KeyError),
        (lambda: gate_matrix(pz.RX), ValueError),
        (lambda: gate_matrix(pz.U2.with_params(0.1)), ValueError),
        (lambda: gate_matrix(pz.U3.with_params(0.1, 0.2)), ValueError),
        (lambda: gate_matrix(pz.CU3.with_params(0.1)), ValueError),
    ],
)
def test_gate_errors(build: object, exc: type[Exception]) -> None:
    """Missing / wrong parameters and unknown names raise."""
    with pytest.raises(exc):
        build()  # type: ignore[operator]


@pytest.mark.parametrize("name", ["rx", "ry", "rz", "p", "u1", "crx", "cp"])
def test_gate_matrix_rejects_extra_params(name: str) -> None:
    """A single-angle gate with two parameters must be rejected."""
    nq = _EXPECTED_PARAM[name][0]
    with pytest.raises(ValueError):
        gate_matrix(Gate(name=name, num_qubits=nq, params=(0.1, 0.2)))
