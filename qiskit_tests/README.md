# qiskit 交叉验证测试套件

`qiskit_tests/` 以 qiskit 为参考实现，对 quspect 的公开接口做交叉验证。最近一次全量运行：
**3231 passed、150 xfailed**，quspect 行覆盖率约 99%。自动生成的详细报告（含分支覆盖率、
逐文件覆盖、xfail 明细）见 [`reports/REPORT.md`](reports/REPORT.md)。

所有已确认缺陷都写成 `@pytest.mark.xfail(strict=True, reason="BUG: ...")`：断言描述的是
**正确行为**，源码修复后用例会 XPASS，并因 `strict=True` 报错，提醒删除 xfail 标记。

---

## 运行方式

### 安装

```bash
pip install -e ".[dev,qiskit-tests]"
pip install pybind11
```

### 构建 C++ 扩展

`Simulator`、`DensityMatrixSimulator`、`MonteCarloSimulator`、`NativeExecutor`、
`AdjointEngine` 依赖 C++ 扩展。需要 C++17 编译器、CMake ≥ 3.22、Ninja 和 pybind11。
**Apple Silicon（arm64）上必须加 `-DQSPECT_BUILD_AVX2=OFF`**：

```bash
cmake -S cpp -B cpp/build -G Ninja \
  -DQSPECT_BUILD_AVX2=OFF \
  -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"
cmake --build cpp/build
```

### 运行测试

```bash
pytest qiskit_tests                       # 全量交叉验证
pytest qiskit_tests -m qiskit_ref         # 只跑与 qiskit 对比的用例
pytest qiskit_tests -m statistical        # 只跑统计采样用例
python -m qiskit_tests.generate_report    # 跑测试 + 分支覆盖率，生成 reports/REPORT.md
python -m qiskit_tests.generate_report --no-run   # 只用已有 junit.xml / coverage.json 重新生成报告
```

注意：默认的 `pytest`（`testpaths = ["tests"]`）**只跑 `tests/`，不会跑 `qiskit_tests`**，
必须显式指定目录。

`generate_report` 的产物都在 `qiskit_tests/reports/`：`REPORT.md`（提交入库），以及
`junit.xml`、`coverage.json`、`htmlcov/`、`.coverage`（已在 `.gitignore` 中忽略）。

---

## 文件结构

| 文件 | 负责范围 |
|---|---|
| `conftest.py` | 共享 fixture（`rng`、`simulator`、`density_simulator`）、marker 注册、约定与 bug 处理规则说明 |
| `helpers.py` | 通用转换、比较、统计容差工具（二项容差、卡方检验、期望值容差） |
| `_sim_helpers.py` | 线路 / 模拟相关测试的辅助函数 |
| `_noise_helpers.py` | 噪声 / 密度矩阵测试用的参考 channel 与工具 |
| `_dag_helpers.py` | DAG / pass / transpiler 测试的辅助函数 |
| `_ir_helpers.py` | IR / 可视化测试的辅助函数 |
| `generate_report.py` | 运行套件并生成 `reports/REPORT.md` |
| `test_conventions.py` | 比特序、门参数顺序等约定以及共享 helper 本身 |
| `test_algorithms_qft.py` | `quspect.algorithms.qft`（`build_qft_circuit` 对比 qiskit QFT） |
| `test_backends_core.py` | `quspect.backends` 与提供方无关的后端层 |
| `test_backends_logicalqubit.py` | `quspect.backends.logicalqubit`（离线，socket 拦截） |
| `test_circuit_api.py` | `quspect.circuit.QuantumCircuit` 构造 API |
| `test_circuit_json.py` | `QuantumCircuit.to_json` / `from_json` 往返 |
| `test_circuit_parameters.py` | `Parameter` / `ParameterExpression` 与参数绑定 |
| `test_dag_phase_polynomial.py` | `quspect.dag.phase_polynomial` |
| `test_dag_structure.py` | `quspect.dag` 的 DAG IR 结构与往返 |
| `test_density_matrix.py` | `quspect.simulation.density_matrix.DensityMatrix` |
| `test_density_simulator.py` | `quspect.simulation.density_simulator.DensityMatrixSimulator` |
| `test_gates_api.py` | `quspect.gates`（`Gate`、模板、`gate_matrix`） |
| `test_ir_compilation.py` | `quspect.ir.compilation` 分解规则 |
| `test_ir_qasm2_export.py` | `quspect.ir.QASM2Exporter`（用 qiskit 解析导出结果） |
| `test_ir_qasm2_import.py` | `quspect.ir.qasm2_loads` / `QASM2Importer` |
| `test_ir_qsal.py` | `quspect.ir.qsal` 序列化与 IR 错误类型 |
| `test_native_executor.py` | `quspect.native` 封装（`NativeExecutor` 等） |
| `test_noise_channels.py` | `quspect.noise.channel` 中所有 channel |
| `test_noise_model.py` | `quspect.noise.noise_model` 与 `quspect.noise.adder` |
| `test_operators_fermion.py` | `quspect.operators.FermionOperator` |
| `test_operators_qubit.py` | `quspect.operators.QubitOperator` |
| `test_operators_transforms.py` | `quspect.operators.transforms`（JW / BK 映射） |
| `test_optimization_conversion.py` | `quspect.optimization` 模型、转换与编译 |
| `test_optimization_expression.py` | `quspect.optimization.expression` 与 `program` |
| `test_optimization_hardware.py` | 硬件降级、CSV codec、native Ising 求解器 |
| `test_passes_optimization.py` | `quspect.dag.passes` 与 `PassManager` 的优化效果 |
| `test_passes_semantics.py` | `quspect.dag.passes` 各 pass 的语义保持 |
| `test_simulation_result.py` | `SimulationResult` 及其构造函数 |
| `test_simulation_sampling.py` | `Simulator.run` counts 的统计验证 |
| `test_simulation_statevector.py` | `Simulator.statevector` / `apply_gate` |
| `test_trajectory_noise.py` | 含噪 Monte Carlo 轨迹模拟的统计验证 |
| `test_transpiler_sabre.py` | `quspect.transpiler.SABRE` 路由 |
| `test_transpiler_target_validation.py` | `CompilationTarget`、transpiler pass 与 `validation` |
| `test_variational_adjoint.py` | `quspect.variational`（objective / adjoint / result） |
| `test_variational_sampling.py` | `quspect.variational.sampling` |
| `test_visualization_sampling.py` | `quspect.visualization.sampling_viz` |
| `test_visualization_svg.py` | `quspect.visualization.svg_drawer` |
| `test_visualization_text.py` | `quspect.visualization.draw_text_circuit` |

---

## 比特序与门约定对照表

| 对象 | quspect 约定 | 与 qiskit 的关系 |
|---|---|---|
| 态矢量 `Simulator().statevector` | 小端，qubit 0 是下标最低位 | 与 qiskit `Statevector` 一致，无需置换 |
| `cx` / `ccx` / `c*` 参数顺序 | 控制位在前：`cx(c,t)`、`ccx(c1,c2,t)`、`crz(θ,c,t)`、`cu3(θ,φ,λ,c,t)` | 与 qiskit 一致；25 个门与 qiskit **完全相等**，没有全局相位差 |
| `gate_matrix(g)`、`NoiseChannel.kraus_operators()` | **大端**：第一个 qubit 参数是最高位 | `gate_matrix(g) == Operator(g).reverse_qargs().data` |
| `apply_gate`、`DensityMatrix.apply_gate` / `apply_channel` | 小端：`qubits[0]` 是最低位 | 与 qiskit `evolve(..., qargs)` 一致 |
| counts 位串 | `s[i]` 是 clbit i（clbit 0 在最左） | 等于 qiskit 位串**反转**：`quspect == qiskit[::-1]` |
| native 原始 counts | MSB-first | 与 qiskit 一致 |
| `QubitOperator` 的 term `((q, P), ...)` | 对应 qiskit label 满足 `s[n-1-q] = P` | `QubitOperator.from_pauli_string(s) == SparsePauliOp(s[::-1])` |
| Jordan–Wigner | `a_j = Z_0 … Z_{j-1} (X_j + iY_j) / 2` | — |
| QFT | `build_qft_circuit(n)` 为正号 DFT，含 swap | 等于 qiskit `QFTGate(n)`，无相位差；`include_swaps=False` 等于 `QFT(do_swaps=False)` |
| SABRE layout | `initial[l]`、`final[l]` 都是逻辑→物理；routed 线路的 wire w 绑定物理比特 `initial[w]` | `U_routed == P_σ (U ⊗ I)`，其中 `σ[l] = initial.index(final[l])` |
| `Depolarizing(p, n)` | `(1-p)ρ + p·I/2^n`；每个非恒等 Pauli 的概率为 `p/4^n` | 与 aer `depolarizing_error(p, n)` 一致 |
| `AmplitudeDamping`、`PhaseDamping` | Nielsen & Chuang 标准形式 | 与 qiskit / aer 一致 |
| `GeneralizedAmplitudeDamping(γ, p_th)` | `p_NC = 1 - p_th` | 换算后与 N&C 形式一致 |
| Ising 编码 | `s = 2x − 1`；`H = offset − Σ h_i Z_i + Σ_{i<j} J_ij Z_i Z_j` | — |

---

## 统计误差尺度

- **单个概率**：5σ 二项容差 `5·sqrt(max(p(1-p), 1/N) / N)`，下限为 `1/N`；单次误报率约 5.7e-7。
- **整体分布**：Pearson 卡方检验，合并期望计数 < 5 的格子，要求 p 值 > 1e-6。
- **期望值**：容差 `5·std/sqrt(N)`；对 ±1 观测量取 `std = sqrt(1 − ⟨Z⟩²)`。
- **shots**：一般测试 4000 shots；噪声轨迹测试 20000 shots（p = 0.5 时 5σ 分辨率约 0.018，
  足以区分 `p/3` 与 `p/4` 这类约定错误）。
- **随机种子**：全部固定 seed；按名字派生 seed 时用 `zlib.crc32`，不用 `hash()`（后者受
  `PYTHONHASHSEED` 影响，不可复现）。
- **无噪声精确比较**：态矢量与幺正矩阵 `atol = 1e-9`；密度矩阵 `1e-10`；单精度 `2e-6`。

---

## 范围与未验证项

- LogicalQubit 云端相关的真实提交、认证、轮询**都没有验证**；测试中用 socket 拦截，保证不会联网。
- 云端 SDK 返回的 counts 位串约定无法离线确认。
- AVX2 内核在 arm64 本地无法编译；由 CI（`.github/workflows/ci.yml`）的 `test (avx2, ubuntu-latest)`
  在 x86_64 runner 上覆盖，并断言运行时实际调度到 `avx2` 内核。scalar 内核由 `test (scalar, macos-latest)` 覆盖。
- 环境中没有 qiskit-aer / qiskit-nature / qiskit-optimization，相应部分改用解析公式或常规方法
  （手写 Kraus、numpy 矩阵、穷举等）验证。

---

## 性能与质量对比（仅作参考）

以下数字只用于观察优化质量的量级，不作为正确性判据。

**Pass 优化后门数**（20 个 motif，共 1208 门）：

| 对比项 | quspect | qiskit |
|---|---:|---:|
| `GateCancellation` vs `InverseCancellation` | 756 | 924 |
| `Commute + Cancel + RotationMerging` vs `CommutativeCancellation` | 375 | 507 |
| `SingleQubitGateFusion` vs `Optimize1qGatesDecomposition` | 672 | 595（quspect 保留了恒等 U3） |

**SABRE 插入的 SWAP 数**：

| 拓扑 | quspect | qiskit |
|---|---:|---:|
| line5 | 58 | 46 |
| ring6 | 56 | 42 |
| grid3x3 | 40 | 24 |
| heavyhex13 | 63 | 35 |

整体约为 qiskit 的 1.4 倍，heavy-hex 上约 1.8 倍。

---

## 缺陷汇总（按严重程度）

> **状态：以下缺陷已全部修复。** 修复记录与逐条状态见 [`BUGFIX_TODO.md`](BUGFIX_TODO.md)。
> 所有原 `xfail(strict=True)` 用例都已转为普通测试，并在 `tests/` 中补了不依赖 qiskit 的回归测试；
> C++ 期望值内核另有 gtest `test_expectation_multi_flip`。本节保留修复前的问题描述，供追溯使用。

每条给出现象、最小复现、根因（若已定位）与对应测试文件。

### 严重：静默给出错误数值

1. **C++ 期望值少一半**
   - 现象：`NativeExecutor.run(return_expectations=True)` 在 Pauli 项含 ≥2 个 X/Y 因子时结果错误。
   - 复现：`|++⟩` 上 `⟨X0 X1⟩` 返回 0.5（应为 1），`⟨Y0 Y1⟩` 返回 -0.5（应为 0）。
   - 根因：`cpp/src/kernels/scalar/expectation.cpp:69-77` 既按 pivot 位过滤、又按 `j <= i` 过滤，
     双重去重导致一半配对丢失；应删除 `j <= i` 判断。`AdjointEngine` 路径不受影响。
   - 测试：`test_ir_qsal.py`
2. **`bravyi_kitaev` 在 n ≥ 3 时违反反对易关系（CAR）**
   - 复现：n = 3 时 `a_0 → σ0 Z1`、`a_2 → σ2` 对易而非反对易；`bravyi_kitaev(a_0 a_2 + a_2 a_0)` 非零；
     JW 与 BK 的谱不一致。
   - 根因：`_bk_ladder_operator` 没有实现 update / parity / remainder 集合。
   - 测试：`test_operators_transforms.py`
3. **`FermionOperator.normal_order` 中 delta 项符号错误**
   - 复现：`a0 a1^ a0^` 得到 `+a1^ + a1^ a0^ a0`，应为 `-a1^ + a1^ a0^ a0`。
   - 根因：后续交换产生的 delta 项使用了原始 `coeff`，而不是更新过符号的 `current_coeff`。
   - 测试：`test_operators_fermion.py`
4. **`FermionOperator.hopping_term(i, i, t)` 只有一半**
   - 复现：返回 `t·n_i`，应为 `2t·n_i`。
   - 根因：dict 字面量中重复 key 被覆盖。
   - 测试：`test_operators_fermion.py`
5. **QASM2 导入时 `.5` 被解析为 5.0**
   - 复现：`rx(.5) q[0];` 导入为 `rx(5.0)`；根因：分词器丢掉了前导小数点。
   - 测试：`test_ir_qasm2_import.py`
6. **QASM2 导入时多个 qreg / creg 相互别名**
   - 复现：`qreg a[1]; qreg b[1];` 中 `b[0] == a[0]`，应按声明顺序拼接；根因：寄存器名被忽略。
   - 测试：`test_ir_qasm2_import.py`
7. **QASM2 导入静默丢弃内容**
   - 复现：`if (c==n) op;` 被静默丢弃（应导入或抛 `QASM2ParseError`）；非法字符被分词器静默跳过。
   - 测试：`test_ir_qasm2_import.py`
8. **`add_thermal_relaxation` 物理模型错误**
   - T2 公式错误：使用 `γ_φ = γ_T2 − γ_T1/2`，相干项不按 `exp(-t/T2)` 衰减（即使 T2 = 2T1 也额外退相位）；
     正确的退相位参数为 `λ = 1 − exp(−2t/T2 + t/T1)`。
   - 标量 T1/T2 被存成 `{0: t}`，只作用于 qubit 0（文档写的是“所有 qubit”）。
   - `excited_state_population` 被忽略，总是向 `|0⟩` 弛豫。
   - 测试：`test_noise_model.py`
9. **`add_readout_error` 从不生效**
   - 根因：规则按 `gate.name == 'measure'` 过滤，但 adder 只遍历 GATE 指令，因此从不插入 channel。
   - 测试：`test_noise_model.py`
10. **`DensityMatrix.probabilities / measure / subsystem_purity` 在部分比特或乱序比特时出错**
    - 复现：3 比特 `|001⟩`（qubit 0 = 1）上 `probabilities((0,))` 返回 `[1, 0, 0, 0]`，应为 `[0, 1]`；
      全排列的 qubit 元组顺序被忽略；`measure` 采样了错误的比特；`subsystem_purity(q)` 返回补集的纯度。
    - 根因：调用 `partial_trace(qubits)`，把要保留的比特迹掉了。
    - 测试：`test_density_matrix.py`、`test_conventions.py`
11. **`DensityMatrixSimulator.run` 结果错误**
    - 忽略 measure → clbit 映射，总是按 qubit 顺序报告全部比特（counts 位数是 num_qubits 而非 num_clbits）。
    - 中途测量只塌缩一次、所有 shot 共用，并且用了上面有缺陷的 `probabilities` 子集，分布错误。
    - 测试：`test_density_simulator.py`、`test_backends_core.py`
12. **`QuantumCircuit.to_json` 丢失噪声强度**
    - 复现：`bit_flip / phase_flip / bit_phase_flip / depolarizing / amplitude_damping / phase_damping`
      的概率或 γ 未序列化，`from_json` 后变为 0，输出分布改变。
    - 测试：`test_circuit_json.py`
13. **QSAL 序列化丢掉观测量复系数的虚部**
    - 复现：`1j*Z0` 往返后变为 0。
    - 测试：`test_ir_qsal.py`
14. **`PhasePolynomialCompiler` / `PhasePolynomial` 语义错误**
    - `extract` 用 `sorted(node.qubits)`，把 `cx(1,0)` 当作 `cx(0,1)`。
    - `extract` 把 `CZ(a,b)` 映射为奇偶 `a⊕b` 上角度 π；正确相位为 `π·ab = π/2·a + π/2·b − π/2·(a⊕b)`。
    - `to_circuit` 在 CNOT 网络之后把每一项都作为 RZ 加在奇偶的第一个 qubit 上，非单位奇偶项相位错误。
    - `optimize()` 用约化形式（单位阵）覆盖奇偶矩阵却不记录 CNOT，破坏语义。
    - 测试：`test_dag_phase_polynomial.py`
15. **`apply_gate` 静默不生效 / 不校验**
    - 对 complex64、float64 或非连续数组静默不生效（C++ 绑定在转换后的副本上操作）。
    - 接受长度不是 2 的幂的态（如长度 3）并修改它，而不是抛 `ValueError`。
    - 测试：`test_simulation_statevector.py`
16. **多比特门重复 qubit 不报错**
    - 复现：`cx(0, 0)` 被接受并被当作恒等门模拟；qiskit 抛 `CircuitError`。
    - 测试：`test_circuit_api.py`

### 中等：合法输入报错、崩溃或挂起

- **噪声 channel 处理**
  - `compose()` 重建指令时丢掉 `channel` 字段，含噪声 channel 的线路 compose 时抛 `ValueError`。（`test_circuit_api.py`）
  - `generalized_amplitude_damp()` 的 channel 名为 `gen_amp_damp`，而 `from_json()` 只认
    `generalized_amplitude_damping`，json 往返抛 `ValueError('Unknown channel type')`。（`test_circuit_json.py`）
  - 噪声模型把 1 比特 channel 匹配到 2 比特门时，作为一条指令插入到门的所有 qubit 上（维度不匹配），
    包括 `thermal_relaxation_model`；所有模拟器（含 native 轨迹解码器）都拒绝执行。
    （`test_noise_model.py`、`test_trajectory_noise.py`）
- **numpy 标量角度**：`np.int64`、`np.float32` 等角度未转换就存储，QSAL JSON 序列化抛 `TypeError`，
  导致无法模拟。（`test_circuit_parameters.py`）
- **QASM2 导入**（`test_ir_qasm2_import.py`）
  - 参数只接受 `[+-]数字` 或 `[+-]pi`，不支持 `pi/2` 等表达式与函数；qiskit `qasm2.dumps` 输出的 `pi/2` 无法读回。
  - 不支持 qelib1.inc / 内建门：`u`、`U`、`CX`、`sx`、`ch`、`cy`、`rzz`、`rxx`、`cu` 等（报 “Unknown gate”）。
  - 不支持整寄存器广播（`h q;`、`measure q -> c;`、`barrier q;`）。
  - 不支持带形式参数的自定义门（`gate g(theta) a {...}`）和自定义门体内调用其它自定义门。
- **QASM2 导出**（`test_ir_qasm2_export.py`）
  - EXTENDED 模式声明带参数的自定义门时没有形式参数（门体硬编码第一个实例的角度），输出非法 QASM。
  - EXTENDED 模式静默丢弃没有 QASM 名的子门，而不是抛 `UnsupportedGateError`。
  - DECOMPOSE 模式只做一层 `decompose_circuit`，嵌套自定义门没有完全展开。
- **SABRE 不连通拓扑**：耦合图不连通（且各分量有边）时循环约 1000 万次 SWAP（约 20 秒）后才抛
  `RuntimeError`，应快速抛 `ValueError`。（`test_transpiler_sabre.py`）
- **`statevector()` 返回空数组**：含中途测量的线路静默返回空数组；空（0 比特）线路返回空数组而不是 `[1]`。
  （`test_simulation_statevector.py`）
- **LogicalQubit**（`test_backends_logicalqubit.py`）
  - `submit_request` 在 `_run_sdk` 已创建远端任务后才校验 `workload_id`，`ValueError` 被包成 `BackendError`，
    已创建的任务既不取消也不上报（孤儿任务）。
  - `LogicalQubitJob(raw_job, return_batch=True)` 不带 labels 时构造 `JobReference(batch_size=0)` 并抛 `ValueError`。
  - `LogicalQubitJob.result` 接受 `timeout=nan`（只检查 `< 0`）。
- **数值边界**
  - `BinaryQuadraticModel` 与 `HardwareIsingProgram` 用 `0.5 * (m + m.T)` 对称化，有限但大于约 8.99e307 的元素
    在通过有限性检查后溢出为 `inf`。（`test_optimization_conversion.py`、`test_optimization_hardware.py`）
  - `AdjointEngine.compile` 不校验非有限角度，native QSAL JSON 解析器抛内部 `RuntimeError` 而不是 `ValueError`。
    （`test_variational_adjoint.py`）
- **PassManager 收敛判断**：只按节点数判断收敛，`CommutationAnalysis` 重排后节点数不变即停止，
  cancel → commute → re-cancel 不会发生（例如 `cx – rz(control) – cx` 被重排却不被消去）。只是优化变差，
  不会产生错误结果。（`test_passes_optimization.py`、`test_transpiler_target_validation.py`）

### 轻微：文档、显示、校验不一致

- `gate_matrix()` 文档写参数个数错误时抛 `ValueError`，实际静默忽略单角度门的多余参数（如 `rx` 传 2 个参数）。（`test_gates_api.py`）
- `CircuitInstruction.matrix()` 文档写符号参数抛 `ValueError`，实际抛 `TypeError`。（`test_circuit_api.py`）
- `bind_parameters()` 不校验类型，字符串等非实数值也被接受。（`test_circuit_parameters.py`）
- `QubitOperator()` 构造不校验：接受非法 Pauli 字母（如 `'Q'`）；保留同一 qubit 出现两次的 term
  （`((0,'X'),(0,'Z'))`），既不拒绝也不化简为 `-1j*Y0`。（`test_operators_qubit.py`）
- `KrausChannel.kraus_ops` 设为 `compare=False`，同名但 Kraus 算符不同的两个 channel 判等为真。（`test_noise_channels.py`）
- 固定单比特 channel 接受 `num_qubits != 1`，并返回与之不符的 2×2 Kraus 算符。（`test_noise_channels.py`）
- `KrausChannel` docstring 示例（`K0=0.9 I, K1=0.1 X`）不保迹（0.81 + 0.01 ≠ 1），照抄会报错。（`test_noise_channels.py`）
- `PassManager.run` 文档承诺不收敛时抛 `RuntimeError`，实际只记录 warning 并返回。（`test_passes_optimization.py`）
- `DAGToCircuitCompiler` 丢弃 CHANNEL（噪声）指令（“unknown instruction type” warning），circuit → DAG → circuit 丢失噪声。（`test_dag_structure.py`）
- QASM2 导出 `_format_float` 只保留 10 位有效数字，导出→导入不能精确往返。（`test_ir_qasm2_export.py`）
- QASM2 导入不校验门的参数个数。（`test_ir_qasm2_import.py`）
- `QASM2ParseError` 消息不含行号（只有 token 或 token 位置）。（`test_ir_qasm2_import.py`）
- QSAL 不保存线路名（`loads()` 后为 `''`）和 `Gate.label`。（`test_ir_qsal.py`）
- 受控旋转门绘图不显示角度：文本绘图画成 `[RZ]/[RX]/[P]/[U3]`，`crz(0.3)` 与 `crz(1.2)` 无法区分；SVG 同样缺角度。
  （`test_visualization_text.py`、`test_visualization_svg.py`）
- 采样可视化（`test_visualization_sampling.py`）
  - DOT 样式按最大计数做除法，0 shots 时抛 `ZeroDivisionError`。
  - `SamplingStats` 把 0 计数条目（`SimulationResult.counts` 中总是存在）算作已观测结果，
    使 `normalized_entropy` 偏低、`num_outcomes` / variance 偏高。
  - VBAR 填充柱宽 6 字符，而空柱和 x 轴标签宽 5 字符，柱子与标签逐渐错位。
- `_repr_latex_` 不省略 ±1 系数：`FermionOperator` 比较 `'1.0'` 而实际得到 `'1'`，`QubitOperator` 正好相反。
  （`test_operators_fermion.py`、`test_operators_qubit.py`）
