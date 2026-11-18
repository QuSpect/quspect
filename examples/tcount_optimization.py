"""用 TCountOptimizePass 降低 Clifford+T 线路的 T 门数。

容错量子计算里 T 门最贵（需要魔术态蒸馏），所以常以 T 门数衡量线路代价。本例构造
有限域 GF(2^m) 乘法器（椭圆曲线密码攻击线路里的基本部件，每个部分积一个 Toffoli），
比较三种设置：

* 默认设置：旋转合并 + TODD；
* ``tensor_walks``：把相位多项式的三次部分看作三线性张量，用翻转图游走找低秩分解
  （相当于自动找到 Karatsuba 式的乘法），以此为起点再做 TODD；
* 再加 ``rm_iterations``：Reed–Muller 大邻域搜索。

运行：``python examples/tcount_optimization.py``（需要已构建原生扩展，否则很慢）。
"""

from __future__ import annotations

import time

import numpy as np

from quspect import QuantumCircuit
from quspect.gates._matrix_utils import gate_matrix
from quspect.transpiler import TCountOptimizePass

#: 各个域使用的不可约多项式（与 Nam 等人的基准线路 gf2^m_mult 相同）。
MODULI = {2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101}


def gf_multiplier(m: int) -> QuantumCircuit:
    """GF(2^m) 上的 ``c ^= a·b``：a、b、c 各 m 个量子比特，每个部分积一个 Toffoli。"""
    modulus = MODULI[m]
    reduced = []
    for degree in range(2 * m - 1):
        value = 1 << degree
        for shift in range(2 * m - 2, m - 1, -1):
            if (value >> shift) & 1:
                value ^= modulus << (shift - m)
        reduced.append(value)
    circuit = QuantumCircuit()
    for i in range(m):
        for j in range(m):
            for k in range(m):
                if (reduced[i + j] >> k) & 1:
                    circuit.ccx(i, m + j, 2 * m + k)
    return circuit


def unitary(circuit: QuantumCircuit, num_qubits: int) -> np.ndarray:
    """线路的幺正矩阵（量子比特 0 为最高位）。"""
    state = np.eye(2**num_qubits, dtype=complex).reshape([2] * num_qubits + [-1])
    for ins in circuit:
        assert ins.gate is not None
        k = len(ins.qubits)
        matrix = gate_matrix(ins.gate).reshape([2] * (2 * k))
        state = np.tensordot(
            matrix, state, axes=(list(range(k, 2 * k)), list(ins.qubits))
        )
        state = np.moveaxis(state, list(range(k)), list(ins.qubits))
    return state.reshape(2**num_qubits, 2**num_qubits)


def equal_up_to_phase(a: np.ndarray, b: np.ndarray) -> bool:
    overlap = np.vdot(b, a)
    return bool(np.max(np.abs(a - overlap / abs(overlap) * b)) < 1e-8)


def main() -> None:
    settings = {
        "default": {},
        "tensor": {"tensor_walks": 16, "tensor_steps": 20_000},
        "tensor+rm": {"tensor_walks": 16, "tensor_steps": 20_000, "rm_iterations": 20},
    }
    header = f"{'circuit':<12}{'setting':<12}{'T in':>6}{'T out':>7}{'CX out':>8}"
    print(f"{header}{'time':>9}  check")
    for m in (2, 3, 4):
        circuit = gf_multiplier(m)
        for name, kwargs in settings.items():
            optimizer = TCountOptimizePass(**kwargs)
            start = time.perf_counter()
            optimized = optimizer.run(circuit)
            elapsed = time.perf_counter() - start
            cx_count = sum(
                1 for ins in optimized if ins.gate is not None and ins.gate.name == "cx"
            )
            # 9 个量子比特以内用稠密矩阵核对（相差一个全局相位）
            check = "-"
            if 3 * m <= 9:
                same = equal_up_to_phase(
                    unitary(circuit, 3 * m), unitary(optimized, 3 * m)
                )
                check = "ok" if same else "MISMATCH"
            stats = optimizer.stats
            print(
                f"gf2^{m}_mult  {name:<12}{stats.t_count_in:>6}{stats.t_count_out:>7}"
                f"{cx_count:>8}{elapsed:>8.2f}s  {check}"
            )


def with_ancillas() -> None:
    """辅助比特版本：Hadamard gadget（干净辅助比特 + 测量 + 经典控制门）。

    线路里夹在 T 门之间的 Hadamard 会把相位多项式切断。``ancillas=True`` 把每个这样的
    Hadamard 换成一个辅助比特、一次测量和按测量结果施加的 Clifford 修正，于是整个
    区域的 T 门合成一个相位多项式一起约简。输出对每一种测量结果都实现同一个幺正
    变换，辅助比特最后回到 |0>。
    """
    circuit = QuantumCircuit()
    for a, b, c in [(0, 1, 3), (2, 3, 4), (0, 1, 3)]:  # 带一个工作比特的三控制 Toffoli
        circuit.ccx(a, b, c)
    plain = TCountOptimizePass()
    plain.run(circuit)
    optimizer = TCountOptimizePass(ancillas=True)
    optimized = optimizer.run(circuit)
    stats = optimizer.stats
    print()
    print(
        f"Toffoli 梯子: T {stats.t_count_in} -> 无辅助比特 {plain.stats.t_count_out}，"
    )
    print(
        f"  有辅助比特 {stats.t_count_out}（辅助比特 {stats.ancillas}，"
        f"测量 {stats.measurements}，经典控制门 {stats.conditional_gates}，"
        f"共 {optimized.num_qubits} 个量子比特）"
    )


if __name__ == "__main__":
    main()
    with_ancillas()
