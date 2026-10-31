"""Energy-equivalence tests for quadratic model conversions."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from quspect.optimization import (
    BinaryQuadraticModel,
    HardwareIsingProgram,
    OptimizationSense,
    VariableType,
    binary_samples_to_spins,
    binary_to_ising,
    bqm_from_qubo_matrix,
    compile_hardware_ising,
    encode_ising_matrix_csv,
    ising_to_binary,
    qubo_matrix_from_bqm,
    spin_samples_to_binary,
)


def _binary_assignments(size: int) -> list[tuple[int, ...]]:
    return list(itertools.product((0, 1), repeat=size))


def _spin_from_binary(bits: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(2 * bit - 1 for bit in bits)


def test_sample_domain_conversion_preserves_shape_and_is_read_only() -> None:
    binary = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int8)
    spins = binary_samples_to_spins(binary)

    np.testing.assert_array_equal(spins, [[-1, 1, -1], [1, 1, -1]])
    np.testing.assert_array_equal(spin_samples_to_binary(spins), binary)
    assert not spins.flags.writeable
    assert not spin_samples_to_binary([-1, 1]).flags.writeable

    with pytest.raises(ValueError, match="0 and 1"):
        binary_samples_to_spins([0, 2])
    with pytest.raises(ValueError, match=r"-1 and \+1"):
        spin_samples_to_binary([-1, 0])


@pytest.mark.parametrize(
    "sense",
    [OptimizationSense.MINIMIZE, OptimizationSense.MAXIMIZE],
)
def test_binary_ising_round_trip_preserves_every_energy(
    sense: OptimizationSense,
) -> None:
    binary = BinaryQuadraticModel(
        variables=("x0", "x1", "x2"),
        linear=np.array([1.25, -2.0, 0.75]),
        quadratic=np.array(
            [
                [0.0, 3.0, -0.5],
                [3.0, 0.0, 1.5],
                [-0.5, 1.5, 0.0],
            ]
        ),
        offset=-4.25,
        variable_type=VariableType.BINARY,
        sense=sense,
    )
    ising = binary_to_ising(binary)
    restored = ising_to_binary(ising)

    for bits in _binary_assignments(binary.num_variables):
        spins = _spin_from_binary(bits)
        expected = binary.evaluate(bits)
        assert ising.evaluate(spins) == pytest.approx(expected)
        assert restored.evaluate(bits) == pytest.approx(expected)
    assert restored.sense is sense


def test_qubo_matrix_normalizes_asymmetric_entries() -> None:
    matrix = np.array(
        [
            [1.0, 2.0, 0.0],
            [3.0, -1.0, 4.0],
            [0.5, 0.0, 2.0],
        ]
    )
    model = bqm_from_qubo_matrix(matrix, offset=0.25)
    canonical = qubo_matrix_from_bqm(model)

    for bits in _binary_assignments(3):
        vector = np.asarray(bits, dtype=np.float64)
        expected = float(vector @ matrix @ vector + 0.25)
        assert model.evaluate(bits) == pytest.approx(expected)
        assert float(vector @ canonical @ vector + model.offset) == pytest.approx(
            expected
        )
    assert np.allclose(canonical, np.triu(canonical))


@pytest.mark.parametrize(
    "sense",
    [OptimizationSense.MINIMIZE, OptimizationSense.MAXIMIZE],
)
@pytest.mark.parametrize("matrix_scale", [0.25, 1.0, 8.0])
def test_hardware_program_preserves_objective_with_gauge(
    sense: OptimizationSense,
    matrix_scale: float,
) -> None:
    binary = BinaryQuadraticModel(
        variables=("x0", "x1"),
        linear=np.array([2.0, -1.0]),
        quadratic=np.array([[0.0, 3.0], [3.0, 0.0]]),
        offset=4.0,
        variable_type=VariableType.BINARY,
        sense=sense,
    )
    ising = binary_to_ising(binary)
    program = compile_hardware_ising(ising, matrix_scale=matrix_scale)

    assert program.gauge_index == 2
    assert program.bit_num == 3
    for bits in _binary_assignments(2):
        logical_spins = np.asarray(_spin_from_binary(bits), dtype=np.int8)
        for gauge in (-1, 1):
            hardware_spins = np.concatenate(
                (logical_spins * gauge, np.array([gauge], dtype=np.int8))
            )
            np.testing.assert_array_equal(
                program.decode_sample(hardware_spins),
                logical_spins,
            )
            assert program.objective_value(hardware_spins) == pytest.approx(
                binary.evaluate(bits)
            )


def test_hardware_program_omits_unneeded_gauge() -> None:
    binary = BinaryQuadraticModel(
        variables=("x0", "x1"),
        linear=np.array([-1.0, -1.0]),
        quadratic=np.array([[0.0, 2.0], [2.0, 0.0]]),
        variable_type=VariableType.BINARY,
    )
    ising = binary_to_ising(binary)
    assert np.allclose(ising.linear, 0.0)

    program = compile_hardware_ising(ising)

    assert program.gauge_index is None
    assert program.bit_num == 2
    assert program.auxiliary_variables == ()


def test_hardware_lowering_matches_documented_qubo_conversion() -> None:
    binary = BinaryQuadraticModel(
        variables=("b1", "b2"),
        linear=np.array([1.0, 1.0]),
        quadratic=np.array([[0.0, 1.0], [1.0, 0.0]]),
        variable_type=VariableType.BINARY,
    )

    program = compile_hardware_ising(binary_to_ising(binary))

    np.testing.assert_allclose(
        program.matrix,
        np.array(
            [
                [0.0, -0.125, -0.375],
                [-0.125, 0.0, -0.375],
                [-0.375, -0.375, 0.0],
            ]
        ),
    )
    assert program.objective_offset == pytest.approx(1.25)


def test_hardware_energy_matches_full_matrix_hamiltonian_convention() -> None:
    matrix = -np.array(
        [
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
        ]
    )
    program = HardwareIsingProgram(
        matrix=matrix,
        variable_order=("s0", "s1", "s2", "s3", "s4"),
        logical_variable_count=5,
    )

    assert program.hardware_energy([-1, -1, -1, 1, 1]) == pytest.approx(-8.0)


def test_hardware_matrix_is_read_only_and_validated() -> None:
    binary = bqm_from_qubo_matrix(np.array([[1.0]]))
    program = compile_hardware_ising(binary_to_ising(binary))

    with pytest.raises(ValueError):
        program.matrix[0, 0] = 1.0
    with pytest.raises(ValueError, match="positive"):
        compile_hardware_ising(binary_to_ising(binary), matrix_scale=0.0)

    near_symmetric = HardwareIsingProgram(
        matrix=np.array([[0.0, 1.0], [1.0 + 5e-13, 0.0]]),
        variable_order=("s0", "s1"),
        logical_variable_count=2,
    )
    assert np.array_equal(near_symmetric.matrix, near_symmetric.matrix.T)
    assert near_symmetric.hardware_energy([1, 1]) == pytest.approx(-(2.0 + 5e-13))


def test_ising_matrix_csv_codec_is_headerless_and_deterministic() -> None:
    binary = BinaryQuadraticModel(
        variables=("x0", "x1"),
        linear=np.array([2.0, -1.0]),
        quadratic=np.array([[0.0, 4.0], [4.0, 0.0]]),
        variable_type=VariableType.BINARY,
    )
    program = compile_hardware_ising(binary_to_ising(binary))

    assert encode_ising_matrix_csv(program) == (
        "0.0,-0.5,-1.0\n-0.5,0.0,-0.25\n-1.0,-0.25,0.0\n"
    )


def test_hardware_program_large_finite_matrix_does_not_overflow() -> None:
    matrix = np.array([[0.0, 1e308], [1e308, 0.0]])
    program = HardwareIsingProgram(matrix, ("a", "b"), logical_variable_count=2)
    np.testing.assert_array_equal(program.matrix, matrix)
