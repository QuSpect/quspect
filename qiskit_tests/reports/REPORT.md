# qiskit 交叉验证测试报告

> 本文件由 `python -m qiskit_tests.generate_report` 自动生成，请勿手工编辑。
> 人工整理的约定、误差尺度与缺陷分级见 [`qiskit_tests/README.md`](../README.md)。

## 运行环境

- 报告生成时间：2026-10-27T18:45:51-07:00
- 测试运行时间戳（junit）：2026-10-27T18:45:02.599925-07:00
- pytest 退出码：0
- Python：3.13.3（macOS-26.5-arm64-arm-64bit-Mach-O）
- qiskit：2.6.1
- quspect git commit：`0757c54`（分支 `test/qiskit-verification`）

## 总览

| 指标 | 数值 |
|---|---|
| 用例总数 | 3381 |
| 通过 | 3381 |
| xfail（已确认缺陷） | 0 |
| 失败 | 0 |
| 错误 | 0 |
| 跳过 | 0 |
| 耗时 | 46.7 s |
| 行覆盖率 | 96.8% (9254/9557) |
| 分支覆盖率 | 93.0% (3335/3586) |
| 标记 `qiskit_ref` 的用例 | 2338 |
| 标记 `statistical` 的用例 | 96 |

去重后的已确认缺陷（按测试文件 + reason）：**0** 条，对应 0 个 xfail 用例。

## 按测试文件

| 测试文件 | 通过 | xfail | 失败 | 错误 | 跳过 | 耗时 (s) |
|---|---:|---:|---:|---:|---:|---:|
| `test_algorithms_qft.py` | 34 | 0 | 0 | 0 | 0 | 0.27 |
| `test_backends_core.py` | 107 | 0 | 0 | 0 | 0 | 0.09 |
| `test_backends_logicalqubit.py` | 153 | 0 | 0 | 0 | 0 | 0.46 |
| `test_circuit_api.py` | 110 | 0 | 0 | 0 | 0 | 0.48 |
| `test_circuit_json.py` | 23 | 0 | 0 | 0 | 0 | 0.06 |
| `test_circuit_parameters.py` | 65 | 0 | 0 | 0 | 0 | 0.05 |
| `test_conventions.py` | 499 | 0 | 0 | 0 | 0 | 1.01 |
| `test_dag_phase_polynomial.py` | 93 | 0 | 0 | 0 | 0 | 0.23 |
| `test_dag_structure.py` | 51 | 0 | 0 | 0 | 0 | 0.42 |
| `test_density_matrix.py` | 70 | 0 | 0 | 0 | 0 | 0.53 |
| `test_density_simulator.py` | 20 | 0 | 0 | 0 | 0 | 0.12 |
| `test_gates_api.py` | 45 | 0 | 0 | 0 | 0 | 0.01 |
| `test_ir_compilation.py` | 119 | 0 | 0 | 0 | 0 | 0.40 |
| `test_ir_qasm2_export.py` | 80 | 0 | 0 | 0 | 0 | 0.26 |
| `test_ir_qasm2_import.py` | 115 | 0 | 0 | 0 | 0 | 0.61 |
| `test_ir_qsal.py` | 65 | 0 | 0 | 0 | 0 | 0.19 |
| `test_native_executor.py` | 21 | 0 | 0 | 0 | 0 | 0.04 |
| `test_noise_channels.py` | 149 | 0 | 0 | 0 | 0 | 0.06 |
| `test_noise_model.py` | 32 | 0 | 0 | 0 | 0 | 0.10 |
| `test_operators_fermion.py` | 46 | 0 | 0 | 0 | 0 | 0.04 |
| `test_operators_qubit.py` | 56 | 0 | 0 | 0 | 0 | 0.08 |
| `test_operators_transforms.py` | 58 | 0 | 0 | 0 | 0 | 0.27 |
| `test_optimization_conversion.py` | 69 | 0 | 0 | 0 | 0 | 0.04 |
| `test_optimization_expression.py` | 57 | 0 | 0 | 0 | 0 | 0.02 |
| `test_optimization_hardware.py` | 87 | 0 | 0 | 0 | 0 | 0.33 |
| `test_passes_optimization.py` | 110 | 0 | 0 | 0 | 0 | 1.03 |
| `test_passes_semantics.py` | 408 | 0 | 0 | 0 | 0 | 4.98 |
| `test_simulation_result.py` | 13 | 0 | 0 | 0 | 0 | 0.02 |
| `test_simulation_sampling.py` | 24 | 0 | 0 | 0 | 0 | 0.29 |
| `test_simulation_statevector.py` | 58 | 0 | 0 | 0 | 0 | 0.05 |
| `test_trajectory_noise.py` | 47 | 0 | 0 | 0 | 0 | 22.69 |
| `test_transpiler_sabre.py` | 146 | 0 | 0 | 0 | 0 | 1.07 |
| `test_transpiler_target_validation.py` | 66 | 0 | 0 | 0 | 0 | 0.14 |
| `test_variational_adjoint.py` | 69 | 0 | 0 | 0 | 0 | 0.30 |
| `test_variational_sampling.py` | 83 | 0 | 0 | 0 | 0 | 4.98 |
| `test_visualization_sampling.py` | 39 | 0 | 0 | 0 | 0 | 0.02 |
| `test_visualization_svg.py` | 44 | 0 | 0 | 0 | 0 | 0.03 |
| `test_visualization_text.py` | 50 | 0 | 0 | 0 | 0 | 0.04 |

## 覆盖率：按子包

| 子包 | 语句 | 行覆盖率 | 分支 | 分支覆盖率 |
|---|---:|---:|---:|---:|
| `circuit` | 647 | 99.5% | 238 | 97.9% |
| `gates` | 202 | 99.5% | 40 | 95.0% |
| `simulation` | 500 | 97.6% | 170 | 92.4% |
| `noise` | 414 | 97.3% | 138 | 94.2% |
| `dag` | 458 | 98.3% | 158 | 92.4% |
| `dag/passes` | 687 | 92.9% | 300 | 86.7% |
| `transpiler` | 609 | 93.9% | 286 | 89.9% |
| `ir` | 1278 | 90.8% | 448 | 82.1% |
| `operators` | 898 | 97.8% | 506 | 97.0% |
| `variational` | 833 | 98.7% | 340 | 96.2% |
| `optimization` | 924 | 99.1% | 286 | 97.9% |
| `backends` | 1220 | 98.6% | 426 | 96.0% |
| `native` | 77 | 97.4% | 2 | 100.0% |
| `algorithms` | 28 | 100.0% | 18 | 100.0% |
| `visualization` | 770 | 99.2% | 230 | 95.2% |
| `(根模块)` | 12 | 100.0% | 0 | — |
| **合计** | 9557 | 96.8% | 3586 | 93.0% |

## 覆盖率：按模块文件

完整的逐行视图见 `qiskit_tests/reports/htmlcov/index.html`。

### `circuit`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/circuit/__init__.py` | 100.0% | — |  |
| `quspect/circuit/circuit.py` | 99.2% | 96.9% | 956, 1093, 1331 |
| `quspect/circuit/instruction.py` | 100.0% | 97.4% |  |
| `quspect/circuit/parameter.py` | 100.0% | 100.0% |  |

### `gates`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/gates/__init__.py` | 100.0% | — |  |
| `quspect/gates/_matrix_utils.py` | 99.1% | 94.4% | 134 |
| `quspect/gates/gate.py` | 100.0% | 100.0% |  |
| `quspect/gates/parameterized.py` | 100.0% | — |  |
| `quspect/gates/standard.py` | 100.0% | — |  |

### `simulation`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/simulation/__init__.py` | 100.0% | — |  |
| `quspect/simulation/density_matrix.py` | 98.4% | 92.0% | 339, 356, 359 |
| `quspect/simulation/density_simulator.py` | 95.9% | 90.0% | 106, 231, 237, 241, 247 |
| `quspect/simulation/gate_applicator.py` | 94.1% | 87.5% | 39, 47 |
| `quspect/simulation/monte_carlo_simulator.py` | 100.0% | — |  |
| `quspect/simulation/result.py` | 100.0% | 100.0% |  |
| `quspect/simulation/simulator.py` | 96.9% | 95.5% | 150-151 |

### `noise`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/noise/__init__.py` | 100.0% | — |  |
| `quspect/noise/adder.py` | 100.0% | 100.0% |  |
| `quspect/noise/channel.py` | 99.1% | 96.2% | 646, 660 |
| `quspect/noise/noise_model.py` | 94.5% | 89.1% | 198, 206, 378, 527, 531, 739, 741-742, 744 |

### `dag`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/dag/__init__.py` | 100.0% | — |  |
| `quspect/dag/compiler.py` | 100.0% | 100.0% |  |
| `quspect/dag/dag_circuit.py` | 100.0% | 97.2% |  |
| `quspect/dag/dag_node.py` | 100.0% | 100.0% |  |
| `quspect/dag/dag_to_circuit.py` | 91.2% | 87.5% | 100, 104, 108 |
| `quspect/dag/phase_polynomial.py` | 97.4% | 89.5% | 339, 351, 382, 400, 474 |

### `dag/passes`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/dag/passes/__init__.py` | 100.0% | — |  |
| `quspect/dag/passes/base.py` | 100.0% | 100.0% |  |
| `quspect/dag/passes/commutation.py` | 92.8% | 85.7% | 118-119, 128, 169, 175, 196, 198, 207, 216, 367, 380 |
| `quspect/dag/passes/diagonal_before_measure.py` | 94.8% | 87.5% | 107, 118, 133 |
| `quspect/dag/passes/gate_cancellation.py` | 88.3% | 82.7% | 74, 98, 174-175, 177-178, 201-202, 213-214, 268, 272, 296 |
| `quspect/dag/passes/rotation_merging.py` | 93.5% | 88.6% | 109-110, 112-113, 172, 181 |
| `quspect/dag/passes/single_qubit_fusion.py` | 90.3% | 84.6% | 75, 230-231, 249, 273, 278, 290, 297, 305, 356, 363, 368-369, 374-375, 386 |

### `transpiler`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/transpiler/__init__.py` | 100.0% | — |  |
| `quspect/transpiler/_mapping.py` | 90.0% | 85.5% | 254, 325, 373, 505, 508-509, 511-524, 526-527, 534-546, 585, 813 |
| `quspect/transpiler/_passes.py` | 100.0% | 100.0% |  |
| `quspect/transpiler/_target.py` | 100.0% | 100.0% |  |
| `quspect/transpiler/validation.py` | 100.0% | 100.0% |  |

### `ir`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/ir/__init__.py` | 100.0% | — |  |
| `quspect/ir/compilation.py` | 100.0% | 96.4% |  |
| `quspect/ir/errors.py` | 100.0% | — |  |
| `quspect/ir/qasm2.py` | 77.6% | 65.4% | 475-477, 528, 599, 647, 650-651, 659, 692-694, 696, 710-711, 736-737, 749, 756-757, 764-767, 769, 793-797, 801-806, 810-813, 815-820, 825, 827-831, 840-842, 852, 855, 857, 863-865, 875-883, 885-889, 891-892 |
| `quspect/ir/qasm_importer.py` | 92.0% | 83.5% | 349-350, 423, 447, 450, 471, 477-478, 500, 525, 534, 574, 622-624, 660, 668, 686, 718, 731, 761, 798, 806-807, 817, 825, 832-836, 839-840, 843, 845, 854-855, 1007, 1021 |
| `quspect/ir/qsal.py` | 99.2% | 97.2% | 529, 723 |

### `operators`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/operators/__init__.py` | 100.0% | — |  |
| `quspect/operators/fermion_operator.py` | 97.9% | 97.4% | 122, 125-126, 623-624, 627-628 |
| `quspect/operators/qubit_operator.py` | 98.3% | 98.4% | 31, 264, 758-762 |
| `quspect/operators/transforms.py` | 96.1% | 90.6% | 231, 356-359, 363 |

### `variational`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/variational/__init__.py` | 100.0% | — |  |
| `quspect/variational/adjoint.py` | 98.2% | 96.3% | 304, 312, 363 |
| `quspect/variational/objective.py` | 100.0% | 100.0% |  |
| `quspect/variational/result.py` | 98.6% | 94.4% | 21 |
| `quspect/variational/sampling.py` | 98.7% | 96.1% | 407, 479, 704, 754, 764, 857, 994 |

### `optimization`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/optimization/__init__.py` | 100.0% | — |  |
| `quspect/optimization/codec.py` | 100.0% | 100.0% |  |
| `quspect/optimization/compiler.py` | 100.0% | 97.7% |  |
| `quspect/optimization/conversion.py` | 100.0% | 100.0% |  |
| `quspect/optimization/expression.py` | 99.2% | 99.0% | 185-186 |
| `quspect/optimization/hardware.py` | 99.2% | 97.7% | 37 |
| `quspect/optimization/model.py` | 99.2% | 96.9% | 82 |
| `quspect/optimization/native_solver.py` | 95.8% | 91.7% | 25-28 |
| `quspect/optimization/program.py` | 100.0% | 100.0% |  |

### `backends`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/backends/__init__.py` | 100.0% | — |  |
| `quspect/backends/_base.py` | 100.0% | 100.0% |  |
| `quspect/backends/_exceptions.py` | 100.0% | — |  |
| `quspect/backends/_execution.py` | 99.3% | 98.1% | 19, 106 |
| `quspect/backends/logicalqubit/__init__.py` | 100.0% | — |  |
| `quspect/backends/logicalqubit/_backend.py` | 99.0% | 95.5% | 192, 479, 547 |
| `quspect/backends/logicalqubit/_job.py` | 94.1% | 86.8% | 182-186, 191-193, 312-314 |
| `quspect/backends/logicalqubit/_provider.py` | 98.9% | 100.0% | 20 |
| `quspect/backends/logicalqubit/_translator.py` | 100.0% | 100.0% |  |

### `native`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/native/__init__.py` | 97.4% | 100.0% | 28-29 |

### `algorithms`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/algorithms/__init__.py` | 100.0% | — |  |
| `quspect/algorithms/qft.py` | 100.0% | 100.0% |  |

### `visualization`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/visualization/__init__.py` | 100.0% | — |  |
| `quspect/visualization/sampling_viz.py` | 99.4% | 96.1% | 202, 801 |
| `quspect/visualization/svg_drawer.py` | 99.6% | 97.4% | 345 |
| `quspect/visualization/text_drawer.py` | 98.4% | 92.3% | 76, 100, 309 |

### `(根模块)`

| 文件 | 行% | 分支% | 未覆盖行 |
|---|---:|---:|---|
| `quspect/__init__.py` | 100.0% | — |  |

## 已确认缺陷清单（xfail）

每条均来自 `xfail(strict=True, reason="BUG: ...")`：断言描述的是正确行为，源码修复后该用例会 XPASS 并因 strict 报错提醒。同一测试文件中 reason 相同的用例（含参数化）合并为一条。
