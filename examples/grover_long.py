"""使用 QuSpect 独立实现并模拟 Grover--Long 精确搜索。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from quspect import QuantumCircuit, Simulator


def calculate_parameters(
    num_search_qubits: int,
    num_marked_states: int,
) -> tuple[int, float]:
    """计算 Grover--Long 的迭代次数和匹配相位。"""
    search_space_size = 1 << num_search_qubits
    beta = math.asin(math.sqrt(num_marked_states / search_space_size))
    iterations = max(1, math.ceil(math.pi / (4 * beta) - 0.5))
    ratio = math.sin(math.pi / (4 * iterations + 2)) / math.sin(beta)
    phase = 2 * math.asin(min(1.0, ratio))
    return iterations, phase


def append_all_ones_phase(
    circuit: QuantumCircuit,
    search_qubits: tuple[int, ...],
    work_qubits: tuple[int, ...],
    phase: float,
) -> None:
    """仅当所有搜索比特为 1 时添加指定相位。"""
    num_search_qubits = len(search_qubits)
    if num_search_qubits == 1:
        circuit.p(phase, search_qubits[0])
        return
    if num_search_qubits == 2:
        circuit.cp(phase, search_qubits[0], search_qubits[1])
        return

    # 用干净辅助比特逐级计算搜索比特的逻辑与。
    circuit.ccx(search_qubits[0], search_qubits[1], work_qubits[0])
    for index in range(2, num_search_qubits - 1):
        circuit.ccx(
            work_qubits[index - 2],
            search_qubits[index],
            work_qubits[index - 1],
        )
    circuit.cp(phase, work_qubits[-1], search_qubits[-1])

    # 逆序还原辅助比特，避免它们与搜索寄存器纠缠。
    for index in range(num_search_qubits - 2, 1, -1):
        circuit.ccx(
            work_qubits[index - 2],
            search_qubits[index],
            work_qubits[index - 1],
        )
    circuit.ccx(search_qubits[0], search_qubits[1], work_qubits[0])


def append_oracle(
    circuit: QuantumCircuit,
    search_qubits: tuple[int, ...],
    work_qubits: tuple[int, ...],
    targets: Sequence[str],
    phase: float,
) -> None:
    """为所有目标态添加相位。"""
    for target in targets:
        zero_qubits = tuple(
            qubit
            for qubit, bit in zip(search_qubits, target, strict=True)
            if bit == "0"
        )
        for qubit in zero_qubits:
            circuit.x(qubit)
        append_all_ones_phase(circuit, search_qubits, work_qubits, phase)
        for qubit in zero_qubits:
            circuit.x(qubit)


def append_diffusion(
    circuit: QuantumCircuit,
    search_qubits: tuple[int, ...],
    work_qubits: tuple[int, ...],
    phase: float,
) -> None:
    """添加具有相同匹配相位的扩散算子。"""
    for qubit in search_qubits:
        circuit.h(qubit)
        circuit.x(qubit)
    append_all_ones_phase(circuit, search_qubits, work_qubits, phase)
    for qubit in search_qubits:
        circuit.x(qubit)
        circuit.h(qubit)


def build_grover_long_circuit(
    targets: Sequence[str],
    *,
    measure: bool,
) -> tuple[QuantumCircuit, int, float]:
    """构造完整的 Grover--Long 搜索线路。"""
    if not targets:
        raise ValueError("至少需要一个目标态")

    num_search_qubits = len(targets[0])
    if num_search_qubits == 0:
        raise ValueError("目标态不能为空")
    if any(
        len(target) != num_search_qubits or set(target) - {"0", "1"}
        for target in targets
    ):
        raise ValueError("目标态必须是等长的二进制字符串")
    if len(set(targets)) != len(targets):
        raise ValueError("目标态不能重复")

    iterations, phase = calculate_parameters(num_search_qubits, len(targets))
    search_qubits = tuple(range(num_search_qubits))
    work_qubits = tuple(range(num_search_qubits, 2 * num_search_qubits - 2))

    circuit = QuantumCircuit(name="grover_long")
    for qubit in search_qubits:
        circuit.h(qubit)

    for _ in range(iterations):
        append_oracle(circuit, search_qubits, work_qubits, targets, phase)
        append_diffusion(circuit, search_qubits, work_qubits, phase)

    if measure:
        circuit.measure(search_qubits, search_qubits)
    return circuit, iterations, phase


def main() -> None:
    """构造三比特搜索线路并输出精确概率与采样结果。"""
    target = "101"

    circuit, iterations, phase = build_grover_long_circuit([target], measure=False)
    simulator = Simulator(seed=42)
    statevector = simulator.statevector(circuit)

    # 辅助比特已还原为零，因此目标态对应的状态向量下标只含搜索比特。
    target_index = sum(int(bit) << qubit for qubit, bit in enumerate(target))
    exact_probability = float(abs(statevector[target_index]) ** 2)

    measured_circuit, _, _ = build_grover_long_circuit([target], measure=True)
    result = simulator.run(measured_circuit, shots=1024)

    print(f"目标态: {target}")
    print(f"迭代次数: {iterations}")
    print(f"匹配相位: {phase:.12f} rad")
    print(f"态矢量目标概率: {exact_probability:.12f}")
    print(f"态矢量归一化: {np.linalg.norm(statevector):.12f}")
    print(f"1024 次采样: {result.counts}")


if __name__ == "__main__":
    main()
