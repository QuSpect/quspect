"""量子隐形传态：用经典控制门（测量结果前馈）完成修正。

Alice 持有待传送的量子比特 q0 和 Bell 对的一半 q1，Bob 持有另一半 q2。Alice 做
Bell 测量得到两个经典比特，Bob 按这两个比特决定是否施加 X、Z：

    c1 == 1  ->  X(q2)
    c0 == 1  ->  Z(q2)

``QuantumCircuit.if_test(clbits, value)`` 块里添加的门只在经典比特等于 ``value``
的那些 shot 上执行。为了核对结果，最后在 q2 上撤销制备操作并测量：传送正确时 q2
一定回到 |0>。

运行：``python examples/teleportation.py``
"""

from __future__ import annotations

from quspect import QuantumCircuit
from quspect.simulation import Simulator
from quspect.transpiler import CompilationTarget, transpile


def teleportation(theta: float, *, correct: bool = True) -> QuantumCircuit:
    """传送 ``ry(theta)|0>``；``correct=False`` 时省略 Bob 的修正。"""
    circuit = QuantumCircuit(name="teleportation")
    circuit.ry(theta, 0)  # 待传送的态
    circuit.h(1).cx(1, 2)  # Bell 对
    circuit.cx(0, 1).h(0)  # Bell 测量的基变换
    circuit.measure([0, 1], [0, 1])
    if correct:
        with circuit.if_test(1, 1):
            circuit.x(2)
        with circuit.if_test(0, 1):
            circuit.z(2)
    circuit.ry(-theta, 2)  # 撤销制备：q2 应回到 |0>
    circuit.measure([2], [2])
    return circuit


def wrong_fraction(circuit: QuantumCircuit, shots: int = 2000) -> float:
    """q2 没有回到 |0> 的 shot 所占比例（计数串从左到右是 c0, c1, c2）。"""
    counts = Simulator(seed=7).run(circuit, shots=shots).counts
    return sum(n for bits, n in counts.items() if bits[2] == "1") / shots


def main() -> None:
    theta = 1.234
    circuit = teleportation(theta)
    print(circuit.draw())
    print(f"有修正:   q2 出错比例 {wrong_fraction(circuit):.3f}")
    uncorrected = wrong_fraction(teleportation(theta, correct=False))
    print(f"没有修正: q2 出错比例 {uncorrected:.3f}")

    # 编译到线形耦合的 {cz, h, rz, x} 设备：条件门保留条件，换成目标门集里的单比特门
    target = CompilationTarget.from_coupling_map(
        3, [(0, 1), (1, 2)], native_gates={"cz", "h", "rz", "x"}
    )
    compiled = transpile(circuit, target, optimization_level=2, seed=0).circuit
    conditional = sum(1 for ins in compiled if ins.is_conditional)
    print(f"编译后:   {len(compiled)} 条指令，其中 {conditional} 条经典控制门，")
    print(f"          q2 出错比例 {wrong_fraction(compiled):.3f}")


if __name__ == "__main__":
    main()
